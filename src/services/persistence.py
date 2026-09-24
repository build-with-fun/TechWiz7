"""Pipeline record -> database rows.  SRS FR lxv-lxxiv, lxxvi, lxxix, lxxx.

The pipeline produces one decision ``record``; this module turns it into the audit trail
that outlives it. Every event is stored with *both* model versions stamped on the row and a
sha256 on the audio file, because an event whose provenance is not recoverable is not an
event you can defend to an evaluator.

Design rules this module honours (from ``src/models.py``):

* ``confidence_scores`` is append-only, one row per (event, model, class), with ``is_top``
  and ``rank`` so the UI and the comparison report never have to re-sort a prediction.
* A ``Review`` row is created *only* when the queue is entered (FR lvii). A pending row on a
  clean result would inflate the queue and misrepresent the model.
* An ``Alert`` row is created only when the pipeline actually raised one (FR liv), carrying
  the rule as it stood, because the rules are editable (FR liii) and "why did this alert
  fire?" must still answer correctly after an edit.
* Audit rows always go through ``record_audit()`` -- the constructor is never called
  directly, and the actor is denormalised so the row outlives its user.
* Nothing here decides anything. The record is the decision; this module is the memory.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.db import StorageLayout, next_audio_id, record_audit
from src.models import (
    Alert,
    AudioFile,
    ConfidenceScore,
    Event,
    ModelVersion,
    Review,
)

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.orm import Session

    from src.models import User

log = logging.getLogger(__name__)

__all__ = ["EventStore", "store_analysis", "resolve_model_version", "make_persistence_callback"]

#: The status the pipeline reports, mapped to the SRS FR lxii status vocabulary the
#: ``events.status`` check constraint admits. The pipeline's own words ("analysed",
#: "rejected", "error") describe *its* progress, not the event's lifecycle.
_EVENT_STATUS: Mapping[str, str] = {
    "analysed": "Classified",
    "error": "Uncertain",
}

#: Map the pipeline's origin word to the two the audio_files check constraint admits.
_AUDIO_SOURCE: Mapping[str, str] = {"upload": "upload", "live": "microphone", "microphone": "microphone"}

_ALLOWED_EVENT_STATUSES = frozenset(
    ("Uploaded", "Classified", "Uncertain", "Alert Generated", "Manual Review",
     "Reviewed", "Closed")
)


# --------------------------------------------------------------------------------------
# Model version registry -- FR lxxv
# --------------------------------------------------------------------------------------

def resolve_model_version(
    session: Session,
    model_name: str,
    *,
    version: str | None,
    feature_version: str | None = None,
    artifact_path: str | None = None,
) -> ModelVersion | None:
    """Find or register the ``ModelVersion`` row a prediction names.

    The event row holds a foreign key to this row, so a decision always points at the model
    that made it. Registering lazily is correct because the pipeline is the only place a
    version number becomes real; a version an evaluator never sees in a prediction is a
    version that was never used. ``is_active`` is set on first registration and re-pointing
    it is the admin API's job, not a side effect of storing an event.
    """
    if not version:
        return None
    name = (model_name or "").strip().lower()
    row = session.execute(
        select(ModelVersion).where(
            ModelVersion.model_name == name, ModelVersion.version == version
        )
    ).scalar_one_or_none()
    if row is not None:
        return row
    row = ModelVersion(
        model_name=name,
        version=version,
        feature_version=feature_version,
        artifact_path=artifact_path,
        is_active=True,
    )
    session.add(row)
    try:
        session.flush()
    except IntegrityError:  # pragma: no cover - raced by another concurrent request
        session.rollback()
        row = session.execute(
            select(ModelVersion).where(
                ModelVersion.model_name == name, ModelVersion.version == version
            )
        ).scalar_one()
    return row


# --------------------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------------------

class EventStore:
    """Applies the record-to-rows mapping inside a session the caller owns.

    A small object rather than a bag of module functions because the audio-id counter and
    the near-duplicate link are per-store state, and because a caller that already has a
    session should not have to reopen it to store one event.
    """

    def __init__(
        self,
        session: Session,
        *,
        storage: StorageLayout | None = None,
        actor: "User | None" = None,
        request_id: str | None = None,
    ) -> None:
        self.session = session
        self.storage = storage or StorageLayout()
        self.actor = actor
        self.request_id = request_id

    # -- helpers --------------------------------------------------------------

    def _audit(self, **kwargs: Any) -> None:
        kwargs.setdefault("request_id", self.request_id)
        record_audit(self.session, actor=self.actor, **kwargs)

    @staticmethod
    def _audio_block(record: Mapping[str, Any]) -> dict[str, Any]:
        audio = dict(record.get("audio") or {})
        if not audio:
            raise ValueError("record has no audio block; nothing to store")
        return audio

    @staticmethod
    def _block(record: Mapping[str, Any], key: str) -> dict[str, Any]:
        return dict(record.get(key) or {})

    @staticmethod
    def _utc() -> datetime:
        # Naive UTC: the schema's DateTime columns are tz-naive and consistent with
        # ``utcnow()`` in models.py. Mixing aware and naive here is what makes the
        # "created_at < x" comparisons in the search API silently misbehave.
        return datetime.now(timezone.utc).replace(tzinfo=None)

    @staticmethod
    def _audio_source(record: Mapping[str, Any]) -> str:
        origin = str(record.get("origin") or record.get("source") or "upload")
        return _AUDIO_SOURCE.get(origin, "upload")

    @staticmethod
    def _size_bytes(record: Mapping[str, Any]) -> int:
        meta = EventStore._block(record, "meta")
        explicit = meta.get("size_bytes")
        if isinstance(explicit, int) and explicit > 0:
            return explicit
        path = EventStore._block(record, "audio").get("source_path")
        if path:
            try:
                return Path(path).stat().st_size
            except OSError:
                pass
        return 0

    @staticmethod
    def _retention_expires(record: Mapping[str, Any]) -> datetime | None:
        days = (EventStore._block(record, "config_snapshot").get("retention") or {}).get(
            "retention_days"
        )
        if not days:
            return None
        try:
            return EventStore._utc() + timedelta(days=int(days))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _review_reason(record: Mapping[str, Any]) -> str | None:
        review = EventStore._block(record, "review")
        if not review.get("required"):
            return None
        findings = review.get("findings") or []
        ids = [str(f.get("id")) for f in findings if isinstance(f, Mapping) and f.get("id")]
        parts = [",".join(ids)] if ids else []
        reason = review.get("reason_text") or review.get("reason")
        if reason:
            parts.append(str(reason))
        return "; ".join(parts) if parts else None

    def _model_version_id(self, model_name: str, block: Mapping[str, Any]) -> int | None:
        version = block.get("model_version")
        if not version:
            return None
        row = resolve_model_version(
            self.session, model_name,
            version=version, feature_version=block.get("feature_version"),
        )
        return None if row is None else row.id

    # -- public ----------------------------------------------------------------

    def store(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Persist one record. Returns a small dict the pipeline echoes into ``stored``.

        Never raises: the pipeline's contract is that a lost write is *reported* on the
        record rather than thrown at the caller, because a user is waiting on the analysis.
        Any failure rolls the whole thing back and comes back as ``{"error": ...}``.
        """
        try:
            return self._store(record)
        except Exception as exc:  # noqa: BLE001 - documented above; the pipeline relies on it
            log.exception("store_analysis failed (status=%s)", record.get("status"))
            try:
                self.session.rollback()
            except Exception:  # pragma: no cover
                pass
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _store(self, record: Mapping[str, Any]) -> dict[str, Any]:
        session = self.session
        audio = self._audio_block(record)
        meta = self._block(record, "meta")

        # -- 1. exact duplicate by content hash (FR lxxiii) ---------------------
        # The pipeline decides whether that is a 409; this is the memory that makes the
        # decision a database guarantee rather than a query that can race.
        existing = self._audio_file_by_hash(audio.get("sha256"))
        if existing is not None:
            self._audit(
                action="audio_duplicate_rejected",
                target_type="audio_file",
                target_id=existing.audio_id,
                outcome="failure",
                detail=f"bytes already stored as {existing.audio_id}",
                after={"audio_id": existing.audio_id, "sha256": existing.sha256},
            )
            # The audit row is added, not flushed; without a commit the refusal itself
            # would vanish from the audit trail, which is the one place it is needed for
            # the duplicate-rejection rate to be explainable.
            self.session.commit()
            return {
                "audio_id": existing.audio_id,
                "duplicate_of": existing.audio_id,
                "event_ids": [],
                "audited": True,
            }

        # -- 2. the audio file, with a fresh id and its bytes on disk ------------
        audio_id = meta.get("audio_id") or next_audio_id(session)
        stored_path = self._write_bytes(record, audio_id=audio_id)
        duplicate = self._block(record, "duplicate")
        near_dup_id = self._near_duplicate_link(duplicate)
        audio_file = self._audio_row(record, audio_id=audio_id, stored_path=stored_path,
                                     near_duplicate_of_id=near_dup_id)
        session.add(audio_file)
        session.flush()  # populate .id and enforce the unique sha256 constraint
        self._audit_audio_upload(audio_file)

        # -- 3. the event (or not) + both models' confidence distributions -------
        event_status = self._event_status(record)
        if event_status is None:
            # A rejected clip is a stored *file*, not an event: it has no prediction, no
            # class and no severity, and an event row would put an empty detection into the
            # timeline and the dashboards. The file is the evidence that it was refused.
            session.commit()
            return {"audio_id": audio_file.audio_id, "event_ids": [], "audited": True}

        predictions = self._block(record, "predictions")
        event = self._event_row(record, audio_file, event_status)
        session.add(event)
        session.flush()
        for score in self._confidence_rows(event, predictions):
            session.add(score)

        # -- 4. the two side effects of the decision: an alert, a review queue entry --
        alert = self._maybe_alert(record, event)
        review = self._maybe_review(record, event)
        if review is not None:
            # The audit row's target is the queue entry itself; the id only exists after
            # the flush, so it is corrected here rather than written twice.
            session.flush()
            self._audit_review_queued(review, event)
        self._audit_prediction(event, record, alert, review)

        # A single commit for the whole record: the audit rows above are written *after*
        # the rows they describe and are added, not flushed, so an early commit here would
        # silently drop them and leave the audit trail missing exactly the row that says
        # what the models concluded.
        session.commit()

        return {
            "audio_id": audio_file.audio_id,
            "event_ids": [event.id],
            "alert_id": None if alert is None else alert.id,
            "review_id": None if review is None else review.id,
            "audited": True,
        }

    # -- the audio file --------------------------------------------------------

    def _audio_file_by_hash(self, sha256: str | None) -> AudioFile | None:
        if not sha256:
            return None
        return self.session.execute(
            select(AudioFile).where(AudioFile.sha256 == sha256)
        ).scalar_one_or_none()

    def _audio_row(
        self,
        record: Mapping[str, Any],
        *,
        audio_id: str,
        stored_path: Path | None,
        near_duplicate_of_id: int | None,
    ) -> AudioFile:
        audio = self._audio_block(record)
        meta = self._block(record, "meta")
        consent = bool(meta.get("consent_acknowledged"))
        return AudioFile(
            audio_id=audio_id,
            filename=str(meta.get("filename") or "recording"),
            stored_path=str(stored_path.relative_to(self.storage.root)) if stored_path else "",
            sha256=audio.get("sha256"),
            perceptual_fingerprint=audio.get("fingerprint"),
            near_duplicate_of_id=near_duplicate_of_id,
            size_bytes=self._size_bytes(record),
            duration_sec=audio.get("duration_sec"),
            sample_rate=audio.get("sample_rate"),
            channels=audio.get("source_channels") or 1,
            container_format=(meta.get("container_format") or "").lower() or None,
            original_format=(meta.get("original_format") or "").lower() or None,
            source=self._audio_source(record),
            location=meta.get("location"),
            created_by_id=getattr(self.actor, "id", None),
            consent_acknowledged=consent,
            consent_recorded_at=self._utc() if consent else None,
            retention_expires_at=self._retention_expires(record),
        )

    def _near_duplicate_link(self, duplicate: Mapping[str, Any]) -> int | None:
        """FR lxxiv: record *which* stored clip this sounds like, without merging them.

        The human decides what a near-duplicate means; the row records the link so the
        decision is available later. The pipeline matched the fingerprint against
        candidates the caller supplied from the database, so the audio_id it names is one
        we already have.
        """
        audio_id = duplicate.get("near_duplicate_of")
        if not audio_id:
            return None
        row = self.session.execute(
            select(AudioFile).where(AudioFile.audio_id == audio_id)
        ).scalar_one_or_none()
        return None if row is None else row.id

    def _write_bytes(self, record: Mapping[str, Any], *, audio_id: str) -> Path | None:
        """Copy the source bytes into storage and return the on-disk path.

        ``stored_path`` is stored relative to the storage root so the database survives the
        project being moved (FR lxvi). When there are no bytes to copy -- an in-memory live
        buffer the client streamed -- the path stays empty and the event keeps its evidence
        in the live-window rows instead.
        """
        audio = self._audio_block(record)
        meta = self._block(record, "meta")
        source = meta.get("source_path") or audio.get("source_path")
        if not source:
            return None
        source_path = Path(source)
        if not source_path.exists():
            return None
        ext = (meta.get("container_format") or source_path.suffix or "").lstrip(".").lower()
        target = self.storage.ensure().audio_path_for(audio_id, ext or "wav")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        return target

    def _audit_audio_upload(self, audio_file: AudioFile) -> None:
        self._audit(
            action="audio_upload",
            target_type="audio_file",
            target_id=audio_file.audio_id,
            detail=f"stored {audio_file.filename} ({audio_file.size_bytes} bytes)",
            after={
                "audio_id": audio_file.audio_id,
                "sha256": audio_file.sha256,
                "fingerprint": audio_file.perceptual_fingerprint,
                "filename": audio_file.filename,
                "size_bytes": audio_file.size_bytes,
                "source": audio_file.source,
            },
        )

    # -- the event -------------------------------------------------------------

    def _event_status(self, record: Mapping[str, Any]) -> str | None:
        """Which FR lxii status the record warrants, or None for "no event row".

        An alert moves the lifecycle to ``Alert Generated`` and a queued review to
        ``Manual Review``; both are the SRS's own words for "the system did something about
        this", which is what the event timeline is for.
        """
        status = record.get("status")
        if status not in _EVENT_STATUS:
            return None
        alert = self._block(record, "alert")
        review = self._block(record, "review")
        # A queued human decision is the current state of the record: whatever else the
        # system did, its outcome is now waiting on a person. An alert that is still
        # unacknowledged is likewise live, but a review supersedes it because the alert's
        # action cannot be taken until the review decides.
        if review.get("required"):
            return "Manual Review"
        if alert.get("raised"):
            return "Alert Generated"
        return _EVENT_STATUS[status]

    def _event_row(
        self, record: Mapping[str, Any], audio_file: AudioFile, status: str
    ) -> Event:
        predictions = self._block(record, "predictions")
        py = self._block(predictions, "python")
        gtm = self._block(predictions, "gtm")
        comparison = self._block(record, "comparison")
        severity = self._block(record, "severity")
        quality = self._block(record, "quality")
        meta = self._block(record, "meta")
        alert = self._block(record, "alert")

        assert status in _ALLOWED_EVENT_STATUSES  # noqa: S101 - a bug, not a user input
        return Event(
            audio_file_id=audio_file.id,
            source=self._audio_source(record),
            status=status,
            predicted_class=py.get("predicted_class"),
            # A review sets final_class; until then the models' output is the record.
            final_class=None,
            severity=severity.get("severity_display") or severity.get("severity"),
            consistency_status=comparison.get("consistency_status"),
            confidence_difference=comparison.get("confidence_difference"),
            top_confidence=py.get("confidence"),
            quality_verdict=quality.get("verdict"),
            quality_score=self._quality_score(quality),
            quality_detail=self._quality_detail(quality),
            python_model_version_id=self._model_version_id("python", py),
            gtm_model_version_id=self._model_version_id("gtm", gtm),
            alert_rule_class=severity.get("rule_class") or severity.get("matched_rule"),
            requires_manual_review=bool(self._block(record, "review").get("required")),
            review_reason=self._review_reason(record),
            config_snapshot=record.get("config_snapshot"),
            overlapping_classes=comparison.get("overlapping_classes"),
            location=meta.get("location"),
            live_session_id=meta.get("live_session_id"),
            consecutive_detections=alert.get("consecutive"),
            classified_at=self._utc(),
            created_by_id=getattr(self.actor, "id", None),
        )

    @staticmethod
    def _quality_score(quality: Mapping[str, Any]) -> float | None:
        measurements = dict(quality.get("measurements") or {})
        for key in ("quality_score", "score", "overall"):
            value = measurements.get(key)
            if isinstance(value, (int, float)):
                return round(float(value), 6)
        return None

    @staticmethod
    def _quality_detail(quality: Mapping[str, Any]) -> str | None:
        """What was wrong with the recording, in the reviewer's own words.

        ``problems`` is the vocabulary the preprocessing module already produced; keeping it
        as-is means the evidence row matches what the pipeline logged.
        """
        problems = list(quality.get("problems") or [])
        summary = quality.get("summary")
        if not problems and not summary:
            return None
        if problems and summary:
            return f"{summary} ({', '.join(problems)})"
        return str(summary or ", ".join(problems) or None) or None

    def _confidence_rows(
        self, event: Event, predictions: Mapping[str, Any]
    ) -> list[ConfidenceScore]:
        """One append-only row per (model, class) -- both models, all classes.

        Storing the whole distribution is what lets the report show the confidence-difference
        *and* the rest of the shape behind it: a 0.04 gap means something different when the
        second-choice class is at 0.30 than when it is at 0.02.
        """
        rows: list[ConfidenceScore] = []
        for model_name in ("python", "gtm"):
            block = self._block(predictions, model_name)
            if not block:
                continue
            top = block.get("predicted_class")
            confidences = dict(block.get("confidences") or {})
            if not confidences and top:
                # A model that reported only its winner still gets a top row, so the
                # per-event evidence is never half-empty.
                confidences = {top: float(block.get("confidence") or 0.0)}
            ordered = sorted(confidences.items(), key=lambda kv: float(kv[1]), reverse=True)
            version_id = self._model_version_id(model_name, block)
            for rank, (class_name, value) in enumerate(ordered, start=1):
                rows.append(
                    ConfidenceScore(
                        event_id=event.id,
                        model_name=model_name,
                        model_version_id=version_id,
                        class_name=class_name,
                        confidence=round(float(value), 6),
                        rank=rank,
                        is_top=class_name == top,
                        latency_sec=block.get("latency_sec"),
                    )
                )
        return rows

    # -- alert and review ------------------------------------------------------

    def _maybe_alert(self, record: Mapping[str, Any], event: Event) -> Alert | None:
        alert = self._block(record, "alert")
        if not alert.get("raised"):
            return None
        severity = self._block(record, "severity")
        row = Alert(
            event_id=event.id,
            severity=severity.get("severity_display") or severity.get("severity") or "Low",
            status="Open",
            rule_class=severity.get("rule_class") or severity.get("matched_rule"),
            rule_snapshot={
                "severity": severity,
                "confirmation": {
                    "consecutive": alert.get("consecutive"),
                    "needed": alert.get("needed"),
                    "window_seconds": alert.get("window_seconds"),
                    "note": alert.get("note"),
                },
                "recommended_action": alert.get("recommended_action"),
            },
            recommended_action=alert.get("recommended_action"),
            message=alert.get("note") or alert.get("message"),
            escalated_to_severity=None,
            dedup_key=self._dedup_key(event),
        )
        self.session.add(row)
        self.session.flush()
        self._audit(
            action="alert_generated",
            target_type="alert",
            target_id=row.id,
            detail=f"{row.severity} alert for {event.predicted_class}",
            after={
                "event_id": event.id,
                "severity": row.severity,
                "rule_class": row.rule_class,
                "sha256": event.audio_file.sha256,
            },
        )
        return row

    @staticmethod
    def _dedup_key(event: Event) -> str | None:
        """Collapse a burst of the same sound into one open alert (FR liv).

        Content hash + class + severity identifies the sound; the key is what makes a repeat
        while the first alert is still open an *update* of that alert instead of a second
        one stacked beneath it.
        """
        cls = event.predicted_class
        if not cls:
            return None
        return f"{event.audio_file.sha256 or ('af' + str(event.audio_file_id))}:{cls}:{event.severity}"

    def _maybe_review(self, record: Mapping[str, Any], event: Event) -> Review | None:
        review = self._block(record, "review")
        if not review.get("required"):
            return None
        predictions = self._block(record, "predictions")
        py = self._block(predictions, "python")
        gtm = self._block(predictions, "gtm")
        severity = self._block(record, "severity")
        findings = review.get("findings") or []
        first_action = next(
            (str(f.get("recommended_action") or "") for f in findings
             if isinstance(f, Mapping) and f.get("recommended_action")),
            "",
        )
        row = Review(
            event_id=event.id,
            status="Pending Review",
            priority=str(review.get("priority") or "normal"),
            condition_ids=[str(c) for c in (review.get("matched") or [])],
            reason_text=self._review_reason(record),
            recommended_action=first_action or None,
            queued_at=self._utc(),
            original_python_class=py.get("predicted_class"),
            original_python_confidence=py.get("confidence"),
            original_gtm_class=gtm.get("predicted_class"),
            original_gtm_confidence=gtm.get("confidence"),
            original_severity=severity.get("severity_display") or severity.get("severity"),
            original_python_model_version=py.get("model_version"),
            original_gtm_model_version=gtm.get("model_version"),
        )
        self.session.add(row)
        return row

    def _audit_review_queued(self, review: Review, event: Event) -> None:
        self._audit(
            action="review_queued",
            target_type="review",
            target_id=review.id,
            detail=self._review_reason({"review": {"required": True, **{}}}) or "manual review required",
            after={
                "event_id": event.id,
                "review_id": review.id,
                "priority": review.priority,
                "condition_ids": review.condition_ids,
                "sha256": event.audio_file.sha256,
            },
        )

    # -- the final audit row ---------------------------------------------------

    def _audit_prediction(
        self,
        event: Event,
        record: Mapping[str, Any],
        alert: Alert | None,
        review: Review | None,
    ) -> None:
        decision = self._block(record, "decision")
        self._audit(
            action="prediction",
            target_type="event",
            target_id=event.id,
            detail=f"{decision.get('final_class') or event.predicted_class} "
                   f"({decision.get('final_decision')})",
            after={
                "event_id": event.id,
                "audio_id": event.audio_file.audio_id,
                "sha256": event.audio_file.sha256,
                "final_class": decision.get("final_class"),
                "final_decision": decision.get("final_decision"),
                "severity": event.severity,
                "consistency_status": event.consistency_status,
                "confidence_difference": event.confidence_difference,
                "requires_manual_review": event.requires_manual_review,
                "alert_id": None if alert is None else alert.id,
                "review_id": None if review is None else review.id,
                "within_budget": record.get("within_budget"),
                "elapsed_ms": record.get("elapsed_ms"),
            },
        )


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------

def store_analysis(
    record: Mapping[str, Any],
    session: Session | None = None,
    *,
    storage: StorageLayout | None = None,
    actor: "User | None" = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Store one pipeline record, opening a session if the caller did not supply one.

    This is the shape the pipeline's ``persist=`` callback expects: the whole record in, a
    small dict out that it copies into ``record["stored"]``.
    """
    if session is not None:
        return EventStore(
            session, storage=storage, actor=actor, request_id=request_id
        ).store(record)

    from src.db import session_scope

    with session_scope() as new_session:
        return EventStore(
            new_session, storage=storage, actor=actor, request_id=request_id
        ).store(record)


def make_persistence_callback(
    storage: StorageLayout | None = None,
    actor: "User | None" = None,
    request_id: str | None = None,
) -> Any:
    """Build the ``persist(record)`` the pipeline calls.

    A factory rather than a bare function so the request's actor and request id are bound
    once at the start of the request instead of being threaded through the pipeline, which
    never needs to see them.
    """

    def _persist(record: Mapping[str, Any]) -> dict[str, Any]:
        return store_analysis(record, storage=storage, actor=actor, request_id=request_id)

    return _persist


def _json_dumps(value: Any) -> Any:  # pragma: no cover - used by tests for round-tripping
    return json.loads(json.dumps(value))
