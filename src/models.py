"""SQLAlchemy models (FR lxxi, lxxii, lxxv, lxxvi, lxxx).

Events point at the model versions that produced them, so a new model doesn't change old
results. Reviewer overrides don't erase model output, and audit rows copy the actor's name
and role so they stay readable after a user is deleted. Datetimes are naive UTC.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Vocabulary shared with the rest of the system

# FR ii. The five roles, stored as these strings.
ROLES = (
    "normal_user",
    "audio_reviewer",
    "security_operator",
    "maintenance_operator",
    "administrator",
)
ROLE_LABELS = {
    "normal_user": "Normal User",
    "audio_reviewer": "Audio Reviewer",
    "security_operator": "Security Operator",
    "maintenance_operator": "Maintenance Operator",
    "administrator": "Administrator",
}

# FR lxii. Event statuses, in the order an event moves through them.
EVENT_STATUSES = (
    "Uploaded",
    "Classified",
    "Uncertain",
    "Alert Generated",
    "Manual Review",
    "Reviewed",
    "Closed",
)

# FR xxxvii.
QUALITY_VERDICTS = ("Good", "Acceptable", "Poor", "Unusable")

# FR xxxiii.
CONSISTENCY_STATUSES = (
    "Strong Match",
    "Acceptable Match",
    "Weak Match",
    "Model Disagreement",
    "Uncertain Result",
)

# FR liii-lvi, plus the final states of a closed alert.
ALERT_STATUSES = ("Open", "Acknowledged", "Dismissed", "Escalated", "Closed")

REVIEW_STATUSES = ("Pending Review", "In Review", "Reviewed")
REVIEW_DECISIONS = ("confirm", "override", "reject", "pending")

# The two models. Fixed at two by the SRS.
MODEL_NAMES = ("python", "gtm")
MODEL_LABELS = {"python": "Python Classification Model", "gtm": "Google Teachable Machine"}

AUDIO_SOURCES = ("upload", "microphone")

# Extension -> MIME type for accepted formats; AudioFile.content_type uses it.
AUDIO_MIME_TYPES = {
    ".wav": "audio/wav",
    ".wave": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/opus",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".webm": "audio/webm",
}


def utcnow() -> _dt.datetime:
    """Current time as naive UTC."""
    return _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)


def to_iso(value: _dt.datetime | None) -> str | None:
    """Serialise a stored UTC datetime as ISO-8601 with a trailing ``Z``."""
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return value.isoformat(timespec="seconds") + "Z"


def sha256_file(path: str) -> str:
    """SHA-256 of a file, read in chunks (FR lxxiii)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Base(DeclarativeBase):
    """Declarative base for all tables."""

    def to_dict(self, *, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for column in self.__table__.columns:
            if column.name in exclude:
                continue
            value = getattr(self, column.name)
            out[column.name] = to_iso(value) if isinstance(value, _dt.datetime) else value
        return out


def _stamp() -> Mapped[_dt.datetime]:
    return mapped_column(DateTime, default=utcnow, nullable=False)


# Users (FR i, FR ii)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    email: Mapped[str | None] = mapped_column(String(255), unique=True)
    display_name: Mapped[str | None] = mapped_column(String(128))
    # PBKDF2-SHA256 hash from werkzeug.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # Lockout is per account, not just per IP (FR i).
    failed_login_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[_dt.datetime | None] = mapped_column(DateTime)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    password_changed_at: Mapped[_dt.datetime | None] = mapped_column(DateTime)
    created_at: Mapped[_dt.datetime] = _stamp()
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    last_login_at: Mapped[_dt.datetime | None] = mapped_column(DateTime)

    audio_files: Mapped[list["AudioFile"]] = relationship(back_populates="created_by")
    events_created: Mapped[list["Event"]] = relationship(back_populates="created_by")

    __table_args__ = (
        CheckConstraint(
            "role IN ('normal_user','audio_reviewer','security_operator',"
            "'maintenance_operator','administrator')",
            name="ck_users_role",
        ),
    )

    @property
    def role_label(self) -> str:
        return ROLE_LABELS.get(self.role, self.role)

    def is_locked(self) -> bool:
        return self.locked_until is not None and self.locked_until > utcnow()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<User {self.username} ({self.role})>"


# Audio (FR lxxi, lxxiii, lxxiv)


class AudioFile(Base):
    """A stored recording. Separate from Event so a file can be re-analysed later."""

    __tablename__ = "audio_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Human-facing identifier used in reports and the UI, e.g. SST-2026-09-23-00004821.
    audio_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    # Relative to the storage root so the data folder can be moved (FR lxxi).
    stored_path: Mapped[str] = mapped_column(String(512), nullable=False)
    # FR lxxiii. Unique, so the database itself rejects exact duplicates.
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    # FR lxxiv. Perceptual fingerprint for near-duplicate matching.
    perceptual_fingerprint: Mapped[str | None] = mapped_column(String(64), index=True)
    near_duplicate_of_id: Mapped[int | None] = mapped_column(ForeignKey("audio_files.id"))

    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_sec: Mapped[float | None] = mapped_column(Float)
    sample_rate: Mapped[int | None] = mapped_column(Integer)
    channels: Mapped[int | None] = mapped_column(Integer)
    # Bits per sample of the uploaded file (FR x); None for lossy formats such as MP3.
    bit_depth: Mapped[int | None] = mapped_column(Integer)
    container_format: Mapped[str | None] = mapped_column(String(16))
    original_format: Mapped[str | None] = mapped_column(String(16))

    source: Mapped[str] = mapped_column(String(16), default="upload", nullable=False, index=True)
    location: Mapped[str | None] = mapped_column(String(255), index=True)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[_dt.datetime] = _stamp()

    # FR lxxix: microphone consent is recorded.
    consent_acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    consent_recorded_at: Mapped[_dt.datetime | None] = mapped_column(DateTime)

    # FR lxxx: when the purge may delete the file. Null means keep forever.
    retention_expires_at: Mapped[_dt.datetime | None] = mapped_column(DateTime, index=True)
    # Legal hold: never purged while set.
    flagged_for_investigation: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)

    created_by: Mapped["User | None"] = relationship(back_populates="audio_files")
    events: Mapped[list["Event"]] = relationship(
        back_populates="audio_file", foreign_keys="Event.audio_file_id"
    )

    __table_args__ = (
        CheckConstraint("source IN ('upload','microphone')", name="ck_audio_source"),
        Index("ix_audio_files_dedupe", "sha256", "perceptual_fingerprint"),
    )

    @property
    def content_type(self) -> str:
        """MIME type for downloads, from the file suffix or the recorded format."""
        suffix = Path(self.stored_path or "").suffix.lower()
        if suffix not in AUDIO_MIME_TYPES:
            recorded = (self.container_format or self.original_format or "").lower().lstrip(".")
            suffix = f".{recorded}" if recorded else ""
        return AUDIO_MIME_TYPES.get(suffix, "application/octet-stream")


# Model versions (FR lxxv)


class ModelVersion(Base):
    """One trained model. Old versions are kept because events refer to them."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str | None] = mapped_column(String(128))
    artifact_path: Mapped[str | None] = mapped_column(String(512))
    feature_version: Mapped[str | None] = mapped_column(String(64))
    # For the report: library, algorithm and scores.
    algorithm: Mapped[str | None] = mapped_column(String(128))
    metrics: Mapped[dict | None] = mapped_column(JSON)
    trained_at: Mapped[_dt.datetime | None] = mapped_column(DateTime)
    registered_at: Mapped[_dt.datetime] = _stamp()
    registered_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    notes: Mapped[str | None] = mapped_column(Text)
    # FR lxxv / SRS 1.8: how this version was trained.
    dataset_manifest_hash: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_version"),
        CheckConstraint("model_name IN ('python','gtm')", name="ck_model_name"),
    )

    @property
    def label_text(self) -> str:
        return MODEL_LABELS.get(self.model_name, self.model_name)


# Events (FR lxii, FR xxxi-xxxiii, FR lxxii)


class Event(Base):
    """One analysed sound event."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    audio_file_id: Mapped[int] = mapped_column(ForeignKey("audio_files.id"), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(16), default="upload", nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(24), default="Uploaded", nullable=False, index=True)

    # what was decided
    predicted_class: Mapped[str | None] = mapped_column(String(64), index=True)
    # Final class after any human override. Null until decided.
    final_class: Mapped[str | None] = mapped_column(String(64), index=True)
    severity: Mapped[str | None] = mapped_column(String(24), index=True)
    # FR xxxiii.
    consistency_status: Mapped[str | None] = mapped_column(String(32), index=True)
    # FR xxxii: |python top confidence - gtm top confidence|.
    confidence_difference: Mapped[float | None] = mapped_column(Float, index=True)
    # Top-class confidence used by the confidence filters.
    top_confidence: Mapped[float | None] = mapped_column(Float, index=True)

    # FR xxxvii audio quality
    quality_verdict: Mapped[str | None] = mapped_column(String(16), index=True)
    quality_score: Mapped[float | None] = mapped_column(Float)
    quality_detail: Mapped[str | None] = mapped_column(Text)

    # FR lxxv: model versions that produced this result
    python_model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), index=True)
    gtm_model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"), index=True)

    # rule and review routing
    alert_rule_class: Mapped[str | None] = mapped_column(String(64))
    requires_manual_review: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    review_reason: Mapped[str | None] = mapped_column(Text)
    # Hashes of the thresholds and rules in force at the time (SRS 1.8 rule 5).
    config_snapshot: Mapped[dict | None] = mapped_column(JSON)
    # FR xxxix: other classes above the overlap threshold.
    overlapping_classes: Mapped[list | None] = mapped_column(JSON)

    location: Mapped[str | None] = mapped_column(String(255), index=True)
    live_session_id: Mapped[str | None] = mapped_column(ForeignKey("live_sessions.id"), index=True)
    # FR xl: number of consecutive windows that confirmed this event.
    consecutive_detections: Mapped[int | None] = mapped_column(Integer)

    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[_dt.datetime] = _stamp()
    classified_at: Mapped[_dt.datetime | None] = mapped_column(DateTime)
    updated_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    audio_file: Mapped["AudioFile"] = relationship(
        back_populates="events", foreign_keys=[audio_file_id]
    )
    created_by: Mapped["User | None"] = relationship(back_populates="events_created")
    python_model_version: Mapped["ModelVersion | None"] = relationship(foreign_keys=[python_model_version_id])
    gtm_model_version: Mapped["ModelVersion | None"] = relationship(foreign_keys=[gtm_model_version_id])
    confidence_scores: Mapped[list["ConfidenceScore"]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )
    alerts: Mapped[list["Alert"]] = relationship(back_populates="event", cascade="all, delete-orphan")
    reviews: Mapped[list["Review"]] = relationship(back_populates="event", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint(
            "status IN ('Uploaded','Classified','Uncertain','Alert Generated',"
            "'Manual Review','Reviewed','Closed')",
            name="ck_event_status",
        ),
        CheckConstraint(
            "quality_verdict IS NULL OR quality_verdict IN ('Good','Acceptable','Poor','Unusable')",
            name="ck_event_quality",
        ),
        CheckConstraint(
            "consistency_status IS NULL OR consistency_status IN "
            "('Strong Match','Acceptable Match','Weak Match','Model Disagreement','Uncertain Result')",
            name="ck_event_consistency",
        ),
        # Used by the event search (FR lxvii).
        Index("ix_events_search", "created_at", "predicted_class", "severity", "status"),
        Index("ix_events_review_queue", "requires_manual_review", "status", "created_at"),
    )

    @property
    def effective_class(self) -> str | None:
        """The reviewer's class if there is one, otherwise the model class."""
        return self.final_class or self.predicted_class

    @property
    def is_critical(self) -> bool:
        return (self.severity or "") == "Critical"

    def audit_ready(self) -> dict[str, Any]:
        """Small JSON-safe summary for audit records."""
        return {
            "id": self.id,
            "status": self.status,
            "predicted_class": self.predicted_class,
            "final_class": self.final_class,
            "severity": self.severity,
            "consistency_status": self.consistency_status,
            "requires_manual_review": self.requires_manual_review,
        }


# Confidence scores (FR lxxii, FR lxxv, FR lxi)


class ConfidenceScore(Base):
    """One row per model, class and event. Never updated. ``is_top`` marks each model's pick."""

    __tablename__ = "confidence_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"))
    class_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    is_top: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    latency_sec: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[_dt.datetime] = _stamp()

    event: Mapped["Event"] = relationship(back_populates="confidence_scores")

    __table_args__ = (
        UniqueConstraint("event_id", "model_name", "class_name", name="uq_confidence_cell"),
        CheckConstraint("model_name IN ('python','gtm')", name="ck_confidence_model"),
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="ck_confidence_range"),
        Index("ix_confidence_top", "event_id", "model_name", "is_top"),
    )


# Alerts (FR liii-lvi)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), default="Open", nullable=False, index=True)
    rule_class: Mapped[str | None] = mapped_column(String(64))
    # Copy of the rule when the alert fired, since rules can be edited (FR liii).
    rule_snapshot: Mapped[dict | None] = mapped_column(JSON)
    recommended_action: Mapped[str | None] = mapped_column(Text)
    message: Mapped[str | None] = mapped_column(Text)
    escalated_to_severity: Mapped[str | None] = mapped_column(String(24))
    # Groups repeated identical events into one alert (FR liv).
    dedup_key: Mapped[str | None] = mapped_column(String(128), index=True)

    created_at: Mapped[_dt.datetime] = _stamp()
    acknowledged_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    acknowledged_at: Mapped[_dt.datetime | None] = mapped_column(DateTime)
    resolved_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    resolved_at: Mapped[_dt.datetime | None] = mapped_column(DateTime)
    # FR lvi: why the alert was dismissed, for the false-alarm rate.
    resolution_note: Mapped[str | None] = mapped_column(Text)
    is_false_alarm: Mapped[bool | None] = mapped_column(Boolean)
    # Notification channels that were tried.
    notified_channels: Mapped[list | None] = mapped_column(JSON)

    event: Mapped["Event"] = relationship(back_populates="alerts")

    __table_args__ = (
        CheckConstraint(
            "status IN ('Open','Acknowledged','Dismissed','Escalated','Closed')",
            name="ck_alert_status",
        ),
        Index("ix_alerts_queue", "status", "severity", "created_at"),
    )

    @property
    def is_open(self) -> bool:
        return self.status in ("Open", "Escalated")


# Reviews (FR lvii-lxi)


class Review(Base):
    """A manual-review item and its outcome, in one row."""

    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), default="Pending Review", nullable=False, index=True)
    priority: Mapped[str] = mapped_column(String(16), default="normal", nullable=False, index=True)
    # FR lvii: conditions that queued this item.
    condition_ids: Mapped[list | None] = mapped_column(JSON)
    reason_text: Mapped[str | None] = mapped_column(Text)
    recommended_action: Mapped[str | None] = mapped_column(Text)
    queued_at: Mapped[_dt.datetime] = _stamp()
    assigned_to_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)

    # the decision (FR lix-lxi)
    decision: Mapped[str] = mapped_column(String(16), default="pending", nullable=False, index=True)
    final_class: Mapped[str | None] = mapped_column(String(64))
    final_severity: Mapped[str | None] = mapped_column(String(24))
    comments: Mapped[str | None] = mapped_column(Text)
    false_alarm: Mapped[bool | None] = mapped_column(Boolean)
    decided_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    decided_at: Mapped[_dt.datetime | None] = mapped_column(DateTime)

    # FR lxi: copy of what the models said, so the review row reads on its own.
    original_python_class: Mapped[str | None] = mapped_column(String(64))
    original_python_confidence: Mapped[float | None] = mapped_column(Float)
    original_gtm_class: Mapped[str | None] = mapped_column(String(64))
    original_gtm_confidence: Mapped[float | None] = mapped_column(Float)
    original_severity: Mapped[str | None] = mapped_column(String(24))
    original_python_model_version: Mapped[str | None] = mapped_column(String(32))
    original_gtm_model_version: Mapped[str | None] = mapped_column(String(32))

    event: Mapped["Event"] = relationship(back_populates="reviews")

    __table_args__ = (
        CheckConstraint(
            "status IN ('Pending Review','In Review','Reviewed')", name="ck_review_status"
        ),
        CheckConstraint(
            "decision IN ('confirm','override','reject','pending')", name="ck_review_decision"
        ),
        Index("ix_reviews_queue", "status", "priority", "queued_at"),
    )

    @property
    def is_decided(self) -> bool:
        return self.status == "Reviewed"


# Live microphone sessions (FR xxxvi, FR lxxix)


class LiveSession(Base):
    __tablename__ = "live_sessions"

    # Random UUID, so ids can't be guessed.
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    device_label: Mapped[str | None] = mapped_column(String(128))
    location: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(16), default="active", nullable=False, index=True)
    started_at: Mapped[_dt.datetime] = _stamp()
    ended_at: Mapped[_dt.datetime | None] = mapped_column(DateTime)
    # FR lxxix: consent given at session start.
    consent_acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    consent_recorded_at: Mapped[_dt.datetime | None] = mapped_column(DateTime)
    window_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    detection_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    alert_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Rolling buffer state per window.
    consecutive_state: Mapped[dict | None] = mapped_column(JSON)

    windows: Mapped[list["LiveWindow"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("status IN ('active','stopped','expired')", name="ck_live_status"),
    )


class LiveWindow(Base):
    """One analysed live window, kept so the detections behind an alert can be checked."""

    __tablename__ = "live_windows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("live_sessions.id"), nullable=False, index=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    captured_at: Mapped[_dt.datetime | None] = mapped_column(DateTime)
    received_at: Mapped[_dt.datetime] = _stamp()
    duration_sec: Mapped[float | None] = mapped_column(Float)

    predicted_class: Mapped[str | None] = mapped_column(String(64))
    python_class: Mapped[str | None] = mapped_column(String(64))
    python_confidence: Mapped[float | None] = mapped_column(Float)
    gtm_class: Mapped[str | None] = mapped_column(String(64))
    gtm_confidence: Mapped[float | None] = mapped_column(Float)
    confidence_difference: Mapped[float | None] = mapped_column(Float)
    consistency_status: Mapped[str | None] = mapped_column(String(32))
    quality_verdict: Mapped[str | None] = mapped_column(String(16))
    severity: Mapped[str | None] = mapped_column(String(24))
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    consecutive: Mapped[int | None] = mapped_column(Integer)
    needed: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    # Set once this window is part of a saved event or alert.
    event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id"), index=True)
    audio_file_id: Mapped[int | None] = mapped_column(ForeignKey("audio_files.id"))

    session: Mapped["LiveSession"] = relationship(back_populates="windows")

    __table_args__ = (
        UniqueConstraint("session_id", "seq", name="uq_live_window_seq"),
    )


# Audit trail (FR lxxvi)


class AuditRecord(Base):
    """Append-only audit record. Keeps the actor's name and role in case the user is deleted."""

    __tablename__ = "audit_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[_dt.datetime] = _stamp()
    # SET NULL so deleting a user keeps their audit rows.
    actor_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    actor_username: Mapped[str | None] = mapped_column(String(64), index=True)
    actor_role: Mapped[str | None] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    target_type: Mapped[str | None] = mapped_column(String(32), index=True)
    target_id: Mapped[str | None] = mapped_column(String(64), index=True)
    outcome: Mapped[str] = mapped_column(String(16), default="success", nullable=False, index=True)
    detail: Mapped[str | None] = mapped_column(Text)
    before: Mapped[dict | None] = mapped_column(JSON)
    after: Mapped[dict | None] = mapped_column(JSON)
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(255))
    # Links the row to the response and the server log.
    request_id: Mapped[str | None] = mapped_column(String(32), index=True)
    sha256: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        Index("ix_audit_when_action", "timestamp", "action"),
        Index("ix_audit_target", "target_type", "target_id"),
    )

    # FR lxxvi actions. The tests check handlers only use these.
    KNOWN_ACTIONS = (
        "login_success", "login_failure", "logout", "account_locked", "password_change",
        "user_create", "user_update", "user_deactivate", "role_change",
        "audio_upload", "audio_duplicate_rejected", "audio_download", "audio_delete",
        "event_flagged", "event_viewed", "evidence_viewed",
        "mic_session_start", "mic_session_stop", "mic_consent",
        "prediction", "prediction_override", "model_version_register", "model_activate",
        "alert_generated", "alert_acknowledged", "alert_dismissed", "alert_escalated",
        "alert_closed", "review_queued", "review_decision", "review_assign",
        "dashboard_view", "search", "export_csv", "export_xlsx", "report_download",
        "config_update", "config_update_rejected", "retention_purge_preview",
        "retention_purge", "anomaly_alert", "access_denied", "request_error",
        "model_failure",
    )
