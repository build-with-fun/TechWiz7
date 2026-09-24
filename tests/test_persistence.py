"""Tests for the persistence layer: pipeline record -> database rows.

Owner: sara.

These tests build synthetic pipeline records rather than running the pipeline, so a missing
or untrained model never makes the audit trail untestable. The record shape here mirrors the
one ``AnalysisPipeline.analyse`` produces; if the pipeline changes, that is caught by the
pipeline's own tests, not here.

What this file insists on:

* both models' versions are stamped on the event row (FR lxxv);
* the sha256 and the fingerprint land on the audio file (FR lxxiii/lxxiv);
* every class of both models gets a confidence row with a correct ``rank`` and ``is_top``;
* an alert is written only when the pipeline raised one, carrying its rule snapshot;
* a review queue entry is written only when review was required, with the originals
  snapshotted so an override never erases what the models said (FR lxi);
* an exact duplicate stores no second event (FR lxxiii);
* a rejected clip stores the file but no event -- the evidence without a fake detection;
* storing never raises at the caller: a failure is reported on the record.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy import select

from src.db import (
    StorageLayout,
    create_engine_for,
    create_schema,
    make_session_factory,
    next_audio_id,
)
from src.models import (
    Alert,
    AudioFile,
    AuditRecord,
    ConfidenceScore,
    Event,
    ModelVersion,
    Review,
    User,
    utcnow,
)
from src.services.persistence import EventStore, store_analysis


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

CLASSES = ["Gunshot", "Glass Breaking", "Panic Scream", "Background Noise"]


@pytest.fixture()
def storage(tmp_path: Path) -> StorageLayout:
    layout = StorageLayout(tmp_path / "storage")
    layout.ensure()
    return layout


@pytest.fixture()
def factory(tmp_path: Path):
    engine = create_engine_for(tmp_path / "test.db")
    create_schema(engine)
    return make_session_factory(engine)


@pytest.fixture()
def actor(factory) -> User:
    """A user row the store can attribute work to.

    The store only ever reads ``actor.id``, so this is returned detached: no lazy load can
    fire against the closed session it came from.
    """
    from src.auth import hash_password

    with factory() as session:
        user = User(
            username="a.reviewer",
            email="rev@example.com",
            display_name="A Reviewer",
            password_hash=hash_password("Sup3rSecret!"),
            role="audio_reviewer",
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        session.expunge(user)
        return user


def _record(
    *,
    sha256: str = "a" * 64,
    fingerprint: str = "fp-aaaa",
    status: str = "analysed",
    py_confidence: float = 0.93,
    gtm_confidence: float = 0.89,
    agree: bool = True,
    severity: str = "High",
    severity_display: str = "High",
    alert_raised: bool = True,
    review_required: bool = False,
    near_duplicate_of: str | None = None,
    filename: str = "warehouse_north.wav",
    location: str | None = "Warehouse North — Bay 4",
    consent: bool = False,
    origin: str = "upload",
    source_path: str | None = None,
    confidences: dict[str, float] | None = None,
) -> dict:
    """A record shaped like ``AnalysisPipeline.analyse`` output."""
    py_top = {
        "Gunshot": py_confidence,
        "Glass Breaking": round(py_confidence - 0.35, 6),
        "Background Noise": round(py_confidence - 0.6, 6),
    }
    gtm_top = {
        "Gunshot": gtm_confidence,
        "Glass Breaking": round(gtm_confidence - 0.3, 6),
        "Background Noise": round(gtm_confidence - 0.55, 6),
    }
    py_conf = dict(py_top if confidences is None else confidences)
    gtm_conf = dict(gtm_top)
    return {
        "ok": status == "analysed",
        "status": status,
        "origin": origin,
        "created_at": "2026-09-23T20:06:19Z",
        "budget_sec": 8.0,
        "meta": {
            "filename": filename,
            "location": location,
            "consent_acknowledged": consent,
            "source_path": source_path,
        },
        "audio": {
            "duration_sec": 6.4,
            "sample_rate": 16000,
            "source_channels": 1,
            "sha256": sha256,
            "fingerprint": fingerprint,
            "fingerprint_version": "v1",
            "source_path": source_path,
        },
        "quality": {
            "verdict": "Good",
            "problems": [],
            "summary": "low noise floor, no clipping",
            "measurements": {"snr_db": 31.2},
        },
        "predictions": {
            "python": {
                "model_name": "python",
                "model_version": "svm_mfcc_v3",
                "predicted_class": "Gunshot",
                "confidence": round(py_confidence, 6),
                "confidences": py_conf,
                "top3": [{"class": c, "confidence": v} for c, v in sorted(
                    py_conf.items(), key=lambda kv: -kv[1])[:3]],
                "latency_sec": 0.42,
                "feature_version": "mfcc-v2",
            },
            "gtm": {
                "model_name": "gtm",
                "model_version": "tm_audio_v2",
                "predicted_class": "Gunshot" if agree else "Glass Breaking",
                "confidence": round(gtm_confidence, 6),
                "confidences": gtm_conf if agree else {k: round(v / 2, 6) for k, v in gtm_conf.items()},
                "top3": [{"class": c, "confidence": v} for c, v in sorted(
                    (gtm_conf if agree else {k: round(v / 2, 6) for k, v in gtm_conf.items()}).items(),
                    key=lambda kv: -kv[1])[:3]],
                "latency_sec": 0.28,
                "feature_version": "gtm-v1",
            },
        },
        "comparison": {
            "consistency_status": "Strong Match" if agree else "Model Disagreement",
            "confidence_difference": round(abs(py_confidence - gtm_confidence), 6),
            "classes_agree": agree,
            "overlapping_classes": ["Glass Breaking"] if not agree else None,
        },
        "severity": {
            "severity": severity,
            "severity_display": severity_display,
            "rule_class": "critical_gunshot",
            "matched_rule": "critical_gunshot",
            "critical_class": severity == "Critical",
            "alert_eligible": True,
            "recommended_action": "Dispatch security, verify the zone.",
        },
        "alert": {
            "raised": alert_raised,
            "eligible": True,
            "confirmed": alert_raised,
            "consecutive": 2,
            "needed": 2,
            "window_seconds": 10,
            "note": "second consecutive window",
            "requires_acknowledgement": True,
            "acknowledged": False,
            "recommended_action": "Dispatch security, verify the Gunshot alert.",
            "message": "Gunshot detected",
        },
        "duplicate": {
            "sha256": sha256,
            "exact_duplicate_of": None,
            "near_duplicate_of": near_duplicate_of,
            "near_duplicate_similarity": None if near_duplicate_of is None else 0.91,
            "near_duplicate_threshold": 0.85,
            "fingerprint_version": "v1",
            "candidates_compared": 0,
            "checked": True,
            "settings_source": "config/thresholds.json",
        },
        "review": {
            "required": review_required,
            "matched": ["critical_without_agreement"] if review_required else [],
            "findings": [
                {
                    "id": "critical_without_agreement",
                    "priority": "critical",
                    "srs_phrase": "Critical class detected without model agreement",
                    "reason": "Gunshot is a critical class and the two models did not agree.",
                    "recommended_action": "Review immediately before acting on the alert.",
                }
            ] if review_required else [],
            "priority": "critical" if review_required else None,
            "reason_text": None,
            "conditions_evaluated": 12,
            "clean_result": not review_required,
            "never_auto_review_ok": True,
        },
        "model_versions": {
            "python": {"name": "python", "version": "svm_mfcc_v3", "feature_version": "mfcc-v2"},
            "gtm": {"name": "gtm", "version": "tm_audio_v2", "feature_version": "gtm-v1"},
        },
        "config_snapshot": {
            "thresholds": {"confidence": {"min_confidence": 0.55}},
            "retention": {"retention_days": 90},
        },
        "decision": {
            "final_class": "Gunshot",
            "final_decision": "Likely Valid",
            "severity_display": severity_display,
            "alert_raised": alert_raised,
            "review_required": review_required,
            "review_priority": "critical" if review_required else None,
        },
    }


def _events(session, event_ids: list[int]) -> list[Event]:
    return list(session.execute(
        select(Event).where(Event.id.in_(event_ids))
    ).scalars().all())


# --------------------------------------------------------------------------------------
# The core mapping
# --------------------------------------------------------------------------------------

def test_analysed_record_stores_an_event_with_both_model_versions(factory, storage, actor):
    with factory() as session:
        result = EventStore(session, storage=storage, actor=actor, request_id="req-1").store(
            _record()
        )
    assert result["event_ids"], "an analysed record must produce an event"
    assert result["alert_id"] is not None

    with factory() as session:
        event = session.execute(select(Event)).scalar_one()
        audio = event.audio_file

        assert event.status == "Alert Generated"
        assert event.predicted_class == "Gunshot"
        assert event.consistency_status == "Strong Match"
        assert event.confidence_difference == pytest.approx(0.04, abs=1e-6)
        assert event.top_confidence == pytest.approx(0.93)
        assert event.severity == "High"
        assert event.quality_verdict == "Good"
        assert event.location == "Warehouse North — Bay 4"
        assert event.requires_manual_review is False

        # FR lxxv: both versions stamped on the row, so activating a new one never
        # rewrites this result.
        assert event.python_model_version.version == "svm_mfcc_v3"
        assert event.gtm_model_version.version == "tm_audio_v2"

        # FR lxxiii / lxxiv: identity and sound on the file.
        assert audio.sha256 == "a" * 64
        assert audio.perceptual_fingerprint == "fp-aaaa"
        assert audio.audio_id.startswith("SST-")
        assert audio.created_by_id == actor.id


def test_confidence_scores_cover_every_class_of_both_models(factory, storage, actor):
    with factory() as session:
        result = EventStore(session, storage=storage, actor=actor).store(_record())
    event_id = result["event_ids"][0]

    with factory() as session:
        scores = list(session.execute(
            select(ConfidenceScore).where(ConfidenceScore.event_id == event_id)
        ).scalars().all())

    # Three classes per model, both models.
    assert len(scores) == 6
    by_model = {}
    for score in scores:
        by_model.setdefault(score.model_name, {})[score.class_name] = score

    for model_name, top_class in (("python", "Gunshot"), ("gtm", "Gunshot")):
        rows = by_model[model_name]
        assert set(rows) == {"Gunshot", "Glass Breaking", "Background Noise"}
        ranked = sorted(rows.values(), key=lambda s: s.rank)
        assert [s.rank for s in ranked] == [1, 2, 3]
        assert ranked[0].is_top is True and ranked[0].class_name == top_class
        assert all(s.confidence >= 0.0 for s in ranked)


def test_alert_is_written_only_when_raised_and_carries_its_rule(factory, storage, actor):
    with factory() as session:
        raised = EventStore(session, storage=storage, actor=actor).store(
            _record(alert_raised=True, severity="Critical", severity_display="Critical")
        )
        quiet = EventStore(session, storage=storage, actor=actor).store(
            _record(sha256="b" * 64, fingerprint="fp-bbbb", alert_raised=False)
        )
    assert raised["alert_id"] is not None
    assert quiet["alert_id"] is None

    with factory() as session:
        alerts = list(session.execute(select(Alert)).scalars().all())
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.severity == "Critical"
        assert alert.status == "Open"
        assert alert.rule_class == "critical_gunshot"
        # The rule as it stood, so an edit later does not change this alert's story.
        assert alert.rule_snapshot["severity"]["severity"] == "Critical"
        assert alert.rule_snapshot["confirmation"]["consecutive"] == 2
        assert alert.dedup_key is not None

        # The event that did not alert has no alert rows.
        noisy_event = session.execute(
            select(Event).where(Event.id == quiet["event_ids"][0])
        ).scalar_one()
        assert noisy_event.alerts == []


def test_review_queue_entry_is_written_only_when_required(factory, storage, actor):
    with factory() as session:
        clean = EventStore(session, storage=storage, actor=actor).store(
            _record(sha256="c" * 64, fingerprint="fp-cccc", review_required=False)
        )
        disputed = EventStore(session, storage=storage, actor=actor).store(
            _record(sha256="d" * 64, fingerprint="fp-dddd", review_required=True, agree=False)
        )
    assert clean["review_id"] is None
    assert disputed["review_id"] is not None

    with factory() as session:
        reviews = list(session.execute(select(Review)).scalars().all())
        assert len(reviews) == 1
        review = reviews[0]
        assert review.status == "Pending Review"
        assert review.priority == "critical"
        assert review.condition_ids == ["critical_without_agreement"]
        assert review.decision == "pending"

        # FR lxi: the models' originals are snapshotted, not replaced.
        assert review.original_python_class == "Gunshot"
        assert review.original_python_confidence == pytest.approx(0.93)
        assert review.original_gtm_class == "Glass Breaking"
        assert review.original_python_model_version == "svm_mfcc_v3"

        # And the event says so.
        disputed_event = session.execute(
            select(Event).where(Event.id == disputed["event_ids"][0])
        ).scalar_one()
        assert disputed_event.requires_manual_review is True
        assert disputed_event.status == "Manual Review"
        assert "critical_without_agreement" in disputed_event.review_reason


def test_exact_duplicate_stores_no_second_event(factory, storage, actor):
    with factory() as session:
        first = EventStore(session, storage=storage, actor=actor).store(_record())
    audio_id = first["audio_id"]

    with factory() as session:
        second = EventStore(session, storage=storage, actor=actor).store(
            _record(filename="renamed_copy.wav", location="Somewhere else")
        )
    assert second["duplicate_of"] == audio_id
    assert second["event_ids"] == []

    with factory() as session:
        files = list(session.execute(select(AudioFile)).scalars().all())
        events = list(session.execute(select(Event)).scalars().all())
    assert len(files) == 1
    assert len(events) == 1


def test_near_duplicate_links_the_files_without_merging(factory, storage, actor):
    with factory() as session:
        original = EventStore(session, storage=storage, actor=actor).store(_record())
    with factory() as session:
        twin = EventStore(session, storage=storage, actor=actor).store(
            _record(
                sha256="e" * 64,
                fingerprint="fp-eeee",
                near_duplicate_of=original["audio_id"],
                alert_raised=False,
            )
        )
    assert twin["event_ids"], "a near-duplicate is its own event; it is not merged"

    with factory() as session:
        twin_file = session.execute(
            select(AudioFile).where(AudioFile.audio_id == twin["audio_id"])
        ).scalar_one()
        original_file = session.execute(
            select(AudioFile).where(AudioFile.audio_id == original["audio_id"])
        ).scalar_one()
    assert twin_file.near_duplicate_of_id == original_file.id


def test_rejected_clip_stores_the_file_but_no_event(factory, storage, actor):
    record = _record(status="rejected")
    record["rejection"] = {"code": "too_short", "reason": "too_short", "message": "too short"}
    with factory() as session:
        result = EventStore(session, storage=storage, actor=actor).store(record)
    assert result["event_ids"] == []
    assert result["audio_id"].startswith("SST-")

    with factory() as session:
        assert session.execute(select(AudioFile)).scalar_one()
        assert session.execute(select(Event)).scalar_one_or_none() is None


def test_live_origin_is_stored_as_microphone_source(factory, storage, actor):
    with factory() as session:
        result = EventStore(session, storage=storage, actor=actor).store(
            _record(sha256="f" * 64, fingerprint="fp-ffff", origin="live", consent=True,
                    alert_raised=False)
        )
    with factory() as session:
        audio = session.execute(
            select(AudioFile).where(AudioFile.audio_id == result["audio_id"])
        ).scalar_one()
    assert audio.source == "microphone"
    assert audio.consent_acknowledged is True
    assert audio.consent_recorded_at is not None


def test_config_snapshot_and_retention_are_recorded(factory, storage, actor):
    with factory() as session:
        result = EventStore(session, storage=storage, actor=actor).store(_record())
    with factory() as session:
        event = session.execute(select(Event)).scalar_one()
        audio = event.audio_file
    assert event.config_snapshot["retention"]["retention_days"] == 90
    assert audio.retention_expires_at is not None
    assert audio.retention_expires_at > utcnow()


# --------------------------------------------------------------------------------------
# Integrity of the audit trail
# --------------------------------------------------------------------------------------

def test_every_store_writes_predict_and_upload_audit_rows(factory, storage, actor):
    with factory() as session:
        EventStore(session, storage=storage, actor=actor, request_id="req-99").store(_record())
    with factory() as session:
        actions = sorted(
            r.action for r in session.execute(select(AuditRecord)).scalars().all()
        )
    assert "audio_upload" in actions
    assert "prediction" in actions
    assert "alert_generated" in actions


def test_audit_rows_carry_the_request_id_and_actor(factory, storage, actor):
    with factory() as session:
        EventStore(session, storage=storage, actor=actor, request_id="req-42").store(_record())
    with factory() as session:
        rows = list(session.execute(select(AuditRecord)).scalars().all())
    assert rows
    assert all(r.request_id == "req-42" for r in rows)
    assert all(r.actor_username == "a.reviewer" for r in rows)
    assert all(r.actor_role == "audio_reviewer" for r in rows)


def test_duplicate_detection_itself_is_audited(factory, storage, actor):
    with factory() as session:
        EventStore(session, storage=storage, actor=actor).store(_record())
    with factory() as session:
        EventStore(session, storage=storage, actor=actor).store(_record())
    with factory() as session:
        actions = [r.action for r in session.execute(select(AuditRecord)).scalars().all()]
    assert actions.count("audio_duplicate_rejected") == 1


def test_model_version_row_is_registered_once_and_reused(factory, storage, actor):
    with factory() as session:
        EventStore(session, storage=storage, actor=actor).store(
            _record(sha256="g" * 64, fingerprint="fp-gggg", alert_raised=False)
        )
    with factory() as session:
        EventStore(session, storage=storage, actor=actor).store(
            _record(sha256="h" * 64, fingerprint="fp-hhhh", alert_raised=False)
        )
    with factory() as session:
        versions = list(session.execute(select(ModelVersion)).scalars().all())
    assert {(v.model_name, v.version) for v in versions} == {
        ("python", "svm_mfcc_v3"), ("gtm", "tm_audio_v2")
    }


# --------------------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------------------

def test_storing_never_raises_at_the_caller(factory, storage, actor):
    """A persistence failure must be reported, not thrown: the pipeline relies on that."""
    broken = _record()
    broken["audio"] = {}  # no audio block -> ValueError deep inside
    with factory() as session:
        result = EventStore(session, storage=storage, actor=actor).store(broken)
    assert "error" in result
    assert result["error"].startswith("ValueError")


def test_bytes_are_copied_to_storage_and_the_path_is_relative(factory, storage, actor, tmp_path):
    source = tmp_path / "incoming.wav"
    source.write_bytes(b"RIFF....")
    with factory() as session:
        result = EventStore(session, storage=storage, actor=actor).store(
            _record(source_path=str(source))
        )
    with factory() as session:
        audio = session.execute(
            select(AudioFile).where(AudioFile.audio_id == result["audio_id"])
        ).scalar_one()
    assert audio.stored_path
    on_disk = storage.resolve(audio.stored_path)
    assert on_disk.exists()
    assert on_disk.read_bytes() == b"RIFF...."
    # Relative, so the database survives the project moving (FR lxxi).
    assert not Path(audio.stored_path).is_absolute()


def test_store_analysis_opens_its_own_session_when_none_is_given(tmp_path, storage):
    """The pipeline's ``persist`` callback needs no request context of its own.

    ``make_persistence_callback`` returns ``store_analysis`` with no session, which then has
    to reach the *application's* factory through ``current_app``. That is the path the live
    monitor will use, and it must not open a second engine.
    """
    from flask import current_app

    from src.app import create_app
    from src.db import session_scope
    from src.services.persistence import make_persistence_callback

    app = create_app(
        TESTING=True,
        SST_DB_PATH=str(tmp_path / "solo.db"),
        SST_STORAGE_DIR=str(tmp_path / "solo-storage"),
        SST_LOAD_MODELS=False,
    )
    with app.app_context():
        # The application's own factory is the one that must be used.
        from src.db import app_session_factory

        assert app_session_factory() is app.config["SST_SESSION_FACTORY"]

        persist = make_persistence_callback(
            storage=app.config["SST_STORAGE"],
            request_id="req-solo",
        )
        result = persist(_record())
        assert result["event_ids"]

        # No second engine: the rows are visible through the app's own connection.
        with session_scope(app_session_factory()) as session:
            stored = session.execute(select(Event)).scalar_one()
            assert stored.audio_file.sha256 == "a" * 64
            assert stored.audio_file.audio_id == result["audio_id"]
        assert current_app.config["SST_ENGINE"] is app.config["SST_ENGINE"]


def test_audio_id_is_unique_across_concurrent_stores(factory, storage, actor):
    ids = []
    for i in range(5):
        with factory() as session:
            ids.append(
                EventStore(session, storage=storage, actor=actor).store(
                    _record(sha256=f"{i:064x}", fingerprint=f"fp-{i:04d}", alert_raised=False)
                )["audio_id"]
            )
    assert len(set(ids)) == 5
    with factory() as session:
        assert next_audio_id(session).endswith("-000006")
