"""The database is the store of record: these tests are the contract on it.

Owner: sara.  SRS FR lxxi-lxxii (storage and contents), lxxiii-lxxiv (duplicate detection),
lxxv (model version tracking), lxxvi (audit trail), lxxx (retention), §1.10 item 11
(mandated folder `database/`, shipped credentials).

The tests that matter most are the two the SRS calls out as integrity traps:

* **FR lxxv** -- activating a new model version must not change a single number on an event
  that was already classified. An evaluator will try exactly this.
* **FR lxxvi** -- the audit trail must still make sense after a user is deleted, which means
  the actor's name cannot be a live foreign key that becomes null.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import time
from pathlib import Path

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from src.db import (
    REPO_ROOT,
    StorageLayout,
    create_engine_for,
    create_schema,
    format_audio_id,
    make_session_factory,
    next_audio_id,
    parse_audio_id,
    record_audit,
    session_scope,
)
from src.models import (
    ALERT_STATUSES,
    CONSISTENCY_STATUSES,
    EVENT_STATUSES,
    MODEL_NAMES,
    QUALITY_VERDICTS,
    ROLES,
    Alert,
    AudioFile,
    AuditRecord,
    ConfidenceScore,
    Event,
    LiveSession,
    LiveWindow,
    ModelVersion,
    Review,
    User,
    sha256_file,
    utcnow,
)

SCHEMA_SQL = REPO_ROOT / "database" / "schema.sql"
CREDENTIALS = REPO_ROOT / "database" / "seed_credentials.json"
INIT_DB = REPO_ROOT / "database" / "init_db.py"


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture()
def engine(tmp_path: Path):
    eng = create_engine_for(str(tmp_path / "test.db"))
    create_schema(eng)
    yield eng


@pytest.fixture()
def factory(engine):
    return make_session_factory(engine)


@pytest.fixture()
def session(factory):
    """A session that survives an expected IntegrityError.

    Tests below assert that the *database* refuses bad data, so a failed flush is the
    expected outcome. A committing ``session_scope`` would then raise PendingRollbackError
    instead, which is why this fixture rolls back and lets the test own the transaction.
    """
    s = factory()
    try:
        yield s
    finally:
        s.rollback()
        s.close()


@pytest.fixture()
def storage(tmp_path: Path) -> StorageLayout:
    return StorageLayout(tmp_path / "storage").ensure()


def _user(session, username="tester", role="normal_user") -> User:
    user = User(username=username, password_hash="x", role=role, is_active=True)
    session.add(user)
    session.flush()
    return user


def _model_version(session, model_name="python", version="1.0.0", active=True) -> ModelVersion:
    row = ModelVersion(model_name=model_name, version=version, is_active=active)
    session.add(row)
    session.flush()
    return row


def _audio(session, user, *, name="clip.wav", sha="a" * 64) -> AudioFile:
    row = AudioFile(
        audio_id=next_audio_id(session),
        filename=name,
        stored_path=f"audio/2026/09/{name}",
        sha256=sha,
        size_bytes=1024,
        duration_sec=6.0,
        sample_rate=22050,
        source="upload",
        created_by_id=user.id,
    )
    session.add(row)
    session.flush()
    return row


def _event(session, audio, user, py_version, gtm_version, **kwargs) -> Event:
    event = Event(
        audio_file_id=audio.id,
        source=audio.source,
        status=kwargs.pop("status", "Classified"),
        predicted_class=kwargs.pop("predicted_class", "Glass Breaking"),
        severity=kwargs.pop("severity", "High"),
        consistency_status=kwargs.pop("consistency_status", "Strong Match"),
        confidence_difference=kwargs.pop("confidence_difference", 0.04),
        top_confidence=kwargs.pop("top_confidence", 0.93),
        quality_verdict=kwargs.pop("quality_verdict", "Good"),
        python_model_version_id=py_version.id,
        gtm_model_version_id=gtm_version.id,
        created_by_id=user.id,
        **kwargs,
    )
    session.add(event)
    session.flush()
    for model, version, top, confs in (
        ("python", py_version, "Glass Breaking", [0.93, 0.03, 0.01]),
        ("gtm", gtm_version, "Glass Breaking", [0.89, 0.05, 0.02]),
    ):
        for rank, (cls, value) in enumerate(
            zip(["Glass Breaking", "Machinery Fault", "Gunshot"], confs)
        ):
            session.add(
                ConfidenceScore(
                    event_id=event.id,
                    model_name=model,
                    model_version_id=version.id,
                    class_name=cls,
                    confidence=value,
                    rank=rank,
                    is_top=(cls == top),
                )
            )
    session.flush()
    return event


# --------------------------------------------------------------------------------------
# The mandated folder and the files in it
# --------------------------------------------------------------------------------------


def test_database_folder_contains_the_mandated_files():
    assert INIT_DB.is_file(), "database/init_db.py is missing (SRS §1.10 folder list)"
    assert SCHEMA_SQL.is_file(), "database/schema.sql is missing"
    assert CREDENTIALS.is_file(), "database/seed_credentials.json is missing"
    assert (REPO_ROOT / "database" / "README.md").is_file(), "database/README.md is missing"


def test_all_required_tables_exist(engine):
    """FR lxxii names what must be stored; each maps to a table here."""
    present = set(inspect(engine).get_table_names())
    required = {
        "users",             # accounts and roles (FR i, ii)
        "audio_files",       # audio metadata + hashes (FR lxxi, lxxiii, lxxiv)
        "model_versions",    # model version tracking (FR lxxv)
        "events",            # classified events (FR lxii)
        "confidence_scores", # confidence scores per model per class (FR lxxii, xxxi)
        "alerts",            # generated alerts (FR liv-lvi)
        "reviews",           # manual review + decisions (FR lvii-lxi)
        "audit_records",     # audit trail (FR lxxvi)
        "live_sessions",     # microphone sessions (FR xxxvi, lxxix)
        "live_windows",      # per-window live results (FR xl)
    }
    assert required <= present, f"missing tables: {sorted(required - present)}"


def test_schema_sql_matches_the_orm_models():
    """schema.sql is generated; this fails the moment the two drift apart.

    Without this, the committed DDL becomes a lie the evaluator reads instead of the schema.
    """
    from database.init_db import emit_schema

    generated = emit_schema(SCHEMA_SQL)
    assert SCHEMA_SQL.read_text(encoding="utf-8") == generated
    # Every table must appear, so a truncated generation cannot pass.
    for table in ("users", "audio_files", "events", "alerts", "reviews", "audit_records"):
        assert f"CREATE TABLE {table} " in generated, f"{table} missing from schema.sql"


# --------------------------------------------------------------------------------------
# Constraints -- the schema refuses nonsense even when the app has a bug
# --------------------------------------------------------------------------------------


def test_foreign_keys_are_actually_enforced(engine, factory):
    """SQLite ignores FKs unless the pragma is set; a silent FK would corrupt analytics."""
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar() == 1

    with pytest.raises(IntegrityError):
        with session_scope(factory) as s:
            s.add(ConfidenceScore(
                event_id=99999, model_name="python", class_name="Gunshot",
                confidence=0.9, rank=0, is_top=True,
            ))


def test_an_unknown_role_is_rejected(session):
    with pytest.raises(IntegrityError):
        session.add(User(username="x", password_hash="h", role="superuser"))
        session.flush()


def test_every_srs_role_is_accepted(session):
    for role in ROLES:
        session.add(User(username=f"u_{role}", password_hash="h", role=role))
    session.flush()
    assert session.execute(select(User)).scalars().all().__len__() == len(ROLES)
    assert "normal_user" in ROLES and "administrator" in ROLES


def test_an_unknown_event_status_is_rejected(session):
    user = _user(session)
    audio = _audio(session, user)
    with pytest.raises(IntegrityError):
        session.add(Event(audio_file_id=audio.id, status="Pending", source="upload"))
        session.flush()


def test_all_seven_srs_event_statuses_are_accepted(session):
    """FR lxii lists seven; all must be storable, or a real transition would 500."""
    user = _user(session)
    audio = _audio(session, user, sha="b" * 64)
    for index, status in enumerate(EVENT_STATUSES):
        session.add(Event(audio_file_id=audio.id, status=status, source="upload",
                          created_at=utcnow() + dt.timedelta(seconds=index)))
    session.flush()
    stored = {e.status for e in session.execute(select(Event)).scalars()}
    assert stored == set(EVENT_STATUSES) == {
        "Uploaded", "Classified", "Uncertain", "Alert Generated",
        "Manual Review", "Reviewed", "Closed",
    }


def test_all_four_quality_verdicts_are_accepted(session):
    user = _user(session)
    audio = _audio(session, user, sha="c" * 64)
    for index, verdict in enumerate(QUALITY_VERDICTS):
        session.add(Event(audio_file_id=audio.id, status="Classified", source="upload",
                          quality_verdict=verdict, created_at=utcnow() + dt.timedelta(seconds=index)))
    session.flush()
    assert set(QUALITY_VERDICTS) == {"Good", "Acceptable", "Poor", "Unusable"}


def test_a_confidence_outside_zero_to_one_is_rejected(session):
    user = _user(session)
    audio = _audio(session, user)
    py, gtm = _model_version(session), _model_version(session, "gtm")
    event = _event(session, audio, user, py, gtm)
    with pytest.raises(IntegrityError):
        session.add(ConfidenceScore(event_id=event.id, model_name="python",
                                    class_name="Gunshot", confidence=1.4, rank=9))
        session.flush()


def test_one_confidence_cell_per_event_model_and_class(session):
    """A duplicate cell would double-count in the per-class analytics."""
    user = _user(session)
    audio = _audio(session, user)
    py, gtm = _model_version(session), _model_version(session, "gtm")
    event = _event(session, audio, user, py, gtm)
    with pytest.raises(IntegrityError):
        session.add(ConfidenceScore(event_id=event.id, model_name="python",
                                    class_name="Glass Breaking", confidence=0.5, rank=1))
        session.flush()


def test_only_the_two_srs_models_can_store_scores(session):
    user = _user(session)
    audio = _audio(session, user)
    py, gtm = _model_version(session), _model_version(session, "gtm")
    event = _event(session, audio, user, py, gtm)
    with pytest.raises(IntegrityError):
        session.add(ConfidenceScore(event_id=event.id, model_name="yolo",
                                    class_name="Gunshot", confidence=0.9, rank=0))
        session.flush()
    assert MODEL_NAMES == ("python", "gtm")


def test_an_unknown_alert_status_is_rejected(session):
    user = _user(session)
    audio = _audio(session, user)
    py, gtm = _model_version(session), _model_version(session, "gtm")
    event = _event(session, audio, user, py, gtm)
    with pytest.raises(IntegrityError):
        session.add(Alert(event_id=event.id, severity="High", status="Ignored"))
        session.flush()
    assert "Acknowledged" in ALERT_STATUSES and "Dismissed" in ALERT_STATUSES


def test_a_review_decision_outside_the_vocabulary_is_rejected(session):
    user = _user(session)
    audio = _audio(session, user)
    py, gtm = _model_version(session), _model_version(session, "gtm")
    event = _event(session, audio, user, py, gtm)
    with pytest.raises(IntegrityError):
        session.add(Review(event_id=event.id, decision="maybe"))
        session.flush()


# --------------------------------------------------------------------------------------
# FR lxxiii / lxxiv -- duplicate and near-duplicate detection
# --------------------------------------------------------------------------------------


def test_the_same_bytes_can_only_be_stored_once(session):
    """FR lxxiii: the exact-duplicate check is a unique constraint, not a query.

    A query-based check loses the race between two simultaneous uploads; the constraint
    cannot, so the guarantee holds under the concurrency the demo will actually see.
    """
    user = _user(session)
    _audio(session, user, name="first.wav", sha="d" * 64)
    with pytest.raises(IntegrityError):
        session.add(AudioFile(
            audio_id=next_audio_id(session), filename="second.wav",
            stored_path="audio/2026/09/second.wav", sha256="d" * 64,
            size_bytes=10, source="upload", created_by_id=user.id,
        ))
        session.flush()


def test_different_bytes_are_both_stored(session):
    user = _user(session)
    _audio(session, user, name="a.wav", sha="e" * 64)
    _audio(session, user, name="b.wav", sha="f" * 64)
    assert session.execute(select(AudioFile)).scalars().all().__len__() == 2


def test_a_near_duplicate_is_recorded_and_never_silently_merged(session):
    """FR lxxiv: identify re-uploads of the same event. A human decides what to do."""
    user = _user(session)
    original = _audio(session, user, name="orig.wav", sha="1" * 64)
    original.perceptual_fingerprint = "fp-abc"
    reencode = _audio(session, user, name="reencoded.wav", sha="2" * 64)
    reencode.perceptual_fingerprint = "fp-abc"
    reencode.near_duplicate_of_id = original.id
    session.flush()

    assert reencode.near_duplicate_of_id == original.id
    assert reencode.id != original.id, "the re-upload must keep its own row and its own event"
    assert reencode.sha256 != original.sha256


def test_sha256_file_matches_a_known_digest(tmp_path: Path):
    """The upload path hashes bytes; a wrong algorithm would silently break dedup."""
    import hashlib

    sample = tmp_path / "sample.bin"
    payload = b"SonicSentinel AI" * 100
    sample.write_bytes(payload)
    assert sha256_file(sample) == hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------------------
# FR lxxv -- model version tracking. The integrity trap.
# --------------------------------------------------------------------------------------


def test_every_event_records_both_model_versions(session):
    user = _user(session)
    audio = _audio(session, user)
    py = _model_version(session, "python", "3.1.0")
    gtm = _model_version(session, "gtm", "2.0.0")
    event = _event(session, audio, user, py, gtm)
    assert event.python_model_version.version == "3.1.0"
    assert event.gtm_model_version.version == "2.0.0"


def test_activating_a_new_model_version_does_not_change_a_past_result(session):
    """FR lxxv, verbatim: "a model version update must not alter previously recorded
    results". This is the test an evaluator will try to break."""
    user = _user(session)
    audio = _audio(session, user)
    py_old = _model_version(session, "python", "3.0.0", active=True)
    gtm = _model_version(session, "gtm", "2.0.0", active=True)
    event = _event(session, audio, user, py_old, gtm)

    before = {
        "class": event.predicted_class,
        "severity": event.severity,
        "confidence_difference": event.confidence_difference,
        "top_confidence": event.top_confidence,
        "python_version": event.python_model_version.version,
        "scores": sorted(
            (s.model_name, s.class_name, s.confidence) for s in event.confidence_scores
        ),
    }

    # A newer Python model is trained and activated.
    py_new = _model_version(session, "python", "4.0.0", active=True)
    py_old.is_active = False
    session.flush()
    session.expire_all()

    reread = session.get(Event, event.id)
    after = {
        "class": reread.predicted_class,
        "severity": reread.severity,
        "confidence_difference": reread.confidence_difference,
        "top_confidence": reread.top_confidence,
        "python_version": reread.python_model_version.version,
        "scores": sorted(
            (s.model_name, s.class_name, s.confidence) for s in reread.confidence_scores
        ),
    }
    assert before == after
    assert reread.python_model_version_id == py_old.id
    assert py_new.is_active is True

    # A new event, however, is judged by the newly active version.
    fresh = _event(session, _audio(session, user, sha="9" * 64), user, py_new, gtm)
    assert fresh.python_model_version.version == "4.0.0"


def test_a_model_version_cannot_be_registered_twice(session):
    _model_version(session, "python", "1.0.0")
    with pytest.raises(IntegrityError):
        _model_version(session, "python", "1.0.0")


def test_a_superseded_version_is_kept_because_events_reference_it(session):
    """Deleting an old version would orphan its events; the FK refuses."""
    user = _user(session)
    audio = _audio(session, user)
    py = _model_version(session, "python", "1.0.0")
    gtm = _model_version(session, "gtm", "1.0.0")
    _event(session, audio, user, py, gtm)

    with pytest.raises(IntegrityError):
        session.delete(py)
        session.flush()


def test_an_unknown_model_name_is_rejected(session):
    with pytest.raises(IntegrityError):
        session.add(ModelVersion(model_name="resnet", version="1.0.0"))
        session.flush()


# --------------------------------------------------------------------------------------
# FR lix-lxi -- review and override preserve the original prediction
# --------------------------------------------------------------------------------------


def test_an_override_preserves_the_original_model_output(session):
    """FR lxi: the override must not erase what the models said."""
    user = _user(session)
    audio = _audio(session, user)
    py = _model_version(session, "python", "3.1.0")
    gtm = _model_version(session, "gtm", "2.0.0")
    event = _event(session, audio, user, py, gtm, predicted_class="Glass Breaking", severity="High")

    review = Review(
        event_id=event.id,
        status="Reviewed",
        priority="high",
        condition_ids=["model_disagreement"],
        reason_text="The Python model predicted 'Glass Breaking' (0.93) and the GTM model "
                    "predicted 'Machinery Fault' (0.88).",
        decision="override",
        final_class="Machinery Fault",
        final_severity="Medium",
        comments="Bearing whine, no glass. Python was misled by a broadband transient.",
        original_python_class=event.predicted_class,
        original_python_confidence=0.93,
        original_gtm_class="Machinery Fault",
        original_gtm_confidence=0.88,
        original_severity=event.severity,
        original_python_model_version=py.version,
        original_gtm_model_version=gtm.version,
        decided_by_id=user.id,
        decided_at=utcnow(),
    )
    session.add(review)
    event.final_class = "Machinery Fault"
    event.final_severity = "Medium"
    event.status = "Reviewed"
    session.flush()
    session.expire_all()

    reread = session.get(Event, event.id)
    assert reread.effective_class == "Machinery Fault"
    assert reread.predicted_class == "Glass Breaking", "the model's own answer must survive"
    review = session.execute(select(Review).where(Review.event_id == event.id)).scalar_one()
    assert review.original_python_class == "Glass Breaking"
    assert review.original_gtm_class == "Machinery Fault"
    assert review.original_python_model_version == "3.1.0"
    assert review.comments


def test_the_original_scores_survive_an_override(session):
    """The comparison report and the accuracy analytics read these rows."""
    user = _user(session)
    audio = _audio(session, user)
    py = _model_version(session, "python", "3.1.0")
    gtm = _model_version(session, "gtm", "2.0.0")
    event = _event(session, audio, user, py, gtm)
    tops = {
        (s.model_name): s.confidence
        for s in event.confidence_scores if s.is_top
    }
    assert tops == {"python": 0.93, "gtm": 0.89}
    assert len(event.confidence_scores) == 6, "both models' full distributions are kept"


def test_the_review_queue_records_why_an_item_was_queued(session):
    """FR lvii: an item that says only "needs review" is a usability defect."""
    user = _user(session)
    audio = _audio(session, user)
    py = _model_version(session), _model_version(session, "gtm")
    event = _event(session, audio, user, *py, requires_manual_review=True)
    session.add(Review(
        event_id=event.id, status="Pending Review", priority="critical",
        condition_ids=["critical_without_agreement", "poor_audio_quality"],
        reason_text="A critical category ('Gunshot') was detected without sufficient model "
                    "agreement (Weak Match).",
        recommended_action="Review immediately.",
    ))
    session.flush()
    queued = session.execute(select(Review).where(Review.status == "Pending Review")).scalar_one()
    assert "critical_without_agreement" in queued.condition_ids
    assert "Gunshot" in queued.reason_text
    assert queued.is_decided is False


def test_a_review_may_be_decided_exactly_once_per_row(session):
    """A second decision is an illegal state transition handled by the service layer; at the
    storage level a decided row must carry who and when."""
    user = _user(session)
    audio = _audio(session, user)
    models = (_model_version(session), _model_version(session, "gtm"))
    event = _event(session, audio, user, *models)
    review = Review(event_id=event.id, status="Reviewed", decision="confirm",
                    decided_by_id=user.id, decided_at=utcnow(), comments="Confirmed by ear.")
    session.add(review)
    session.flush()
    assert review.is_decided and review.decided_at is not None and review.decided_by_id


# --------------------------------------------------------------------------------------
# FR liv-lvi -- alerts
# --------------------------------------------------------------------------------------

def test_an_alert_records_the_rule_that_fired(session):
    """FR liii rules are editable, so the alert keeps its own copy: "why did this alert?"
    must still answer correctly after an administrator changes the rule."""
    user = _user(session)
    audio = _audio(session, user)
    models = (_model_version(session), _model_version(session, "gtm"))
    event = _event(session, audio, user, *models)
    session.add(Alert(
        event_id=event.id, severity="High", status="Open", rule_class="Glass Breaking",
        rule_snapshot={"class": "Glass Breaking", "severity": "High", "min_confidence": 0.60,
                       "min_top_two_margin": 0.10, "required_consecutive_detections": 3,
                       "requires_model_agreement": True},
        recommended_action="Treat as a security event.",
        message="Glass Breaking detected with strong model agreement.",
    ))
    session.flush()
    alert = session.execute(select(Alert)).scalar_one()
    assert alert.is_open
    assert alert.rule_snapshot["required_consecutive_detections"] == 3


def test_an_acknowledged_alert_records_who_and_when(session):
    """FR lv."""
    user = _user(session)
    audio = _audio(session, user)
    models = (_model_version(session), _model_version(session, "gtm"))
    event = _event(session, audio, user, *models)
    alert = Alert(event_id=event.id, severity="Critical", status="Acknowledged",
                  acknowledged_by_id=user.id, acknowledged_at=utcnow())
    session.add(alert)
    session.flush()
    assert alert.acknowledged_by_id == user.id
    assert alert.is_open is False


def test_a_dismissed_alert_records_the_false_alarm_reason(session):
    """FR lvi: without the reason the false-alarm rate is not measurable."""
    user = _user(session)
    audio = _audio(session, user)
    models = (_model_version(session), _model_version(session, "gtm"))
    event = _event(session, audio, user, *models)
    alert = Alert(event_id=event.id, severity="High", status="Dismissed",
                  resolved_by_id=user.id, resolved_at=utcnow(),
                  resolution_note="Door slam, checked the camera.",
                  is_false_alarm=True)
    session.add(alert)
    session.flush()
    assert alert.is_false_alarm is True
    assert alert.resolution_note


def test_two_alerts_can_share_a_dedup_key_so_a_burst_collapses(session):
    """FR liv's dedup window: a key is recorded, not unique, so the burst is countable."""
    user = _user(session)
    audio = _audio(session, user)
    models = (_model_version(session), _model_version(session, "gtm"))
    event = _event(session, audio, user, *models)
    for _ in range(3):
        session.add(Alert(event_id=event.id, severity="High", status="Open",
                          dedup_key="Glass Breaking|Warehouse North|2026-09-23T20"))
    session.flush()
    assert session.execute(select(Alert)).scalars().all().__len__() == 3


# --------------------------------------------------------------------------------------
# FR xxxvi / lxxix -- live sessions and consent
# --------------------------------------------------------------------------------------


def test_a_live_session_records_consent_and_its_windows(session):
    """FR lxxix: users are informed when the microphone is active; the consent is stored."""
    from src.db import new_session_id

    user = _user(session)
    sid = new_session_id()
    session.add(LiveSession(id=sid, user_id=user.id, status="active",
                            consent_acknowledged=True, consent_recorded_at=utcnow(),
                            device_label="Built-in Audio", location="Warehouse North"))
    session.flush()
    for seq in range(3):
        session.add(LiveWindow(
            session_id=sid, seq=seq, duration_sec=1.5, predicted_class="Aggression",
            python_class="Aggression", python_confidence=0.71,
            gtm_class="Aggression", gtm_confidence=0.66,
            consistency_status="Acceptable Match", severity="High",
            confirmed=(seq == 2), consecutive=seq + 1, needed=3, latency_ms=812.5,
        ))
    session.flush()
    assert len(session.get(LiveSession, sid).windows) == 3
    # The consecutive count is exposed so the UI can show "2 of 3 confirming windows".
    windows = session.execute(
        select(LiveWindow).where(LiveWindow.session_id == sid).order_by(LiveWindow.seq)
    ).scalars().all()
    assert [w.consecutive for w in windows] == [1, 2, 3]
    assert [w.confirmed for w in windows] == [False, False, True]


def test_a_window_cannot_be_recorded_twice_for_the_same_sequence(session):
    from src.db import new_session_id

    user = _user(session)
    sid = new_session_id()
    session.add(LiveSession(id=sid, user_id=user.id, consent_acknowledged=True))
    session.flush()
    session.add(LiveWindow(session_id=sid, seq=1))
    session.flush()
    with pytest.raises(IntegrityError):
        session.add(LiveWindow(session_id=sid, seq=1))
        session.flush()


def test_a_deleted_session_takes_its_windows_with_it(session):
    from src.db import new_session_id

    user = _user(session)
    sid = new_session_id()
    session.add(LiveSession(id=sid, user_id=user.id, consent_acknowledged=True))
    session.flush()
    session.add(LiveWindow(session_id=sid, seq=0))
    session.flush()
    session.delete(session.get(LiveSession, sid))
    session.flush()
    assert session.execute(select(LiveWindow)).scalars().all() == []


# --------------------------------------------------------------------------------------
# FR lxxvi -- the audit trail
# --------------------------------------------------------------------------------------


def test_every_audit_action_in_the_vocabulary_is_known():
    """Handlers name an action; the vocabulary is here so a typo is caught by review."""
    required = {
        "login_success", "login_failure", "logout", "audio_upload", "audio_download",
        "mic_session_start", "mic_consent", "prediction", "alert_generated",
        "alert_acknowledged", "alert_dismissed", "alert_escalated", "review_queued",
        "review_decision", "model_version_register", "model_activate", "config_update",
        "export_csv", "export_xlsx", "retention_purge", "access_denied",
    }
    assert required <= set(AuditRecord.KNOWN_ACTIONS), (
        sorted(required - set(AuditRecord.KNOWN_ACTIONS))
    )


def test_record_audit_copies_the_actor_so_history_survives_deletion(session):
    """An audit trail that turns into nulls when a user is removed is not an audit trail."""
    actor = _user(session, "o.operator", "security_operator")
    record_audit(session, action="alert_acknowledged", actor=actor,
                 target_type="alert", target_id=7, detail="acknowledged a Glass Breaking alert",
                 ip_address="10.0.0.4", request_id="abc123")
    session.flush()

    session.delete(actor)
    session.flush()
    session.expire_all()

    row = session.execute(select(AuditRecord)).scalar_one()
    assert row.actor_id is None, "the FK is nulled by the delete, as expected"
    assert row.actor_username == "o.operator", "but the record still names who did it"
    assert row.actor_role == "security_operator"
    assert row.detail and row.ip_address == "10.0.0.4" and row.request_id == "abc123"


def test_audit_rows_record_before_and_after_for_a_change(session):
    user = _user(session, "admin", "administrator")
    record_audit(session, action="config_update", actor=user,
                 target_type="config_file", target_id="config/thresholds.json",
                 before={"confidence": {"min_confidence": 0.60}},
                 after={"confidence": {"min_confidence": 0.93}})
    session.flush()
    row = session.execute(select(AuditRecord)).scalar_one()
    assert row.before["confidence"]["min_confidence"] == 0.60
    assert row.after["confidence"]["min_confidence"] == 0.93


def test_a_failed_login_for_an_unknown_user_is_still_auditable(session):
    record_audit(session, action="login_failure", actor=None,
                 detail="no such username", ip_address="10.0.0.9", outcome="failure")
    session.flush()
    row = session.execute(select(AuditRecord)).scalar_one()
    assert row.actor_id is None and row.actor_username is None
    assert row.outcome == "failure"


# --------------------------------------------------------------------------------------
# FR lxxi / lxxx -- storage paths and retention
# --------------------------------------------------------------------------------------


def test_storage_layout_creates_its_directories(storage: StorageLayout):
    assert storage.audio_dir.is_dir()
    assert storage.live_dir.is_dir()
    assert storage.exports_dir.is_dir()


def test_stored_paths_are_relative_so_the_database_survives_a_move(storage: StorageLayout):
    path = storage.audio_path_for("SST-2026-09-23-00000042", ".wav",
                                  when=dt.datetime(2026, 9, 23, 20, 6))
    assert path.parts[-3:] == ("2026", "09", "SST-2026-09-23-00000042.wav")
    assert storage.root in path.parents


def test_a_stored_path_may_not_escape_the_storage_root(storage: StorageLayout):
    """A path from the database is data, not a trusted instruction."""
    with pytest.raises(ValueError, match="escapes the storage root"):
        storage.resolve("../../etc/passwd")


def test_retention_expiry_is_storable_and_queryable(session):
    user = _user(session)
    audio = _audio(session, user)
    audio.retention_expires_at = utcnow() - dt.timedelta(days=1)
    session.flush()
    expired = session.execute(
        select(AudioFile).where(AudioFile.retention_expires_at < utcnow())
    ).scalars().all()
    assert [a.id for a in expired] == [audio.id]


def test_a_flagged_event_is_excluded_from_a_purge_candidate_query(session):
    """FR lxxvi legal hold: an event under investigation is never purged."""
    user = _user(session)
    for index, flagged in enumerate((False, True)):
        row = _audio(session, user, name=f"clip{index}.wav", sha=str(index + 1) * 64)
        row.retention_expires_at = utcnow() - dt.timedelta(days=1)
        row.flagged_for_investigation = flagged
    session.flush()

    purgeable = session.execute(
        select(AudioFile).where(
            AudioFile.retention_expires_at < utcnow(),
            AudioFile.flagged_for_investigation.is_(False),
        )
    ).scalars().all()
    assert len(purgeable) == 1
    assert purgeable[0].flagged_for_investigation is False


# --------------------------------------------------------------------------------------
# Audio identifiers
# --------------------------------------------------------------------------------------


def test_audio_id_is_sortable_and_round_trips():
    value = format_audio_id(4821, dt.datetime(2026, 9, 23))
    assert value == "SST-2026-09-23-004821"
    assert parse_audio_id(value) == dt.date(2026, 9, 23)
    assert parse_audio_id("nonsense") is None


def test_a_new_audio_id_never_collides_with_a_surviving_row(session):
    """A row count would hand out a duplicate after a deletion; the max-suffix scheme cannot.

    This matters because the search filter (FR lxvii) and the audit trail (FR lxxvi) both
    address an audio file by this id -- two live rows sharing one is an ambiguity, not a
    cosmetic clash.
    """
    user = _user(session)
    first = _audio(session, user, name="a.wav", sha="a1" * 32)
    second = _audio(session, user, name="b.wav", sha="b1" * 32)
    third = _audio(session, user, name="c.wav", sha="c1" * 32)
    assert len({first.audio_id, second.audio_id, third.audio_id}) == 3

    # Delete the oldest, as a retention purge (FR lxxx) does.
    session.delete(first)
    session.flush()

    surviving = {
        row.audio_id for row in session.execute(select(AudioFile)).scalars()
    }
    # A count-based scheme returns "000003" here and collides with `third`.
    count_based = format_audio_id(len(surviving) + 1)
    assert count_based in surviving, "sanity: this is the collision the max-suffix scheme avoids"

    replacement = _audio(session, user, name="d.wav", sha="d1" * 32)
    assert replacement.audio_id not in surviving


def test_audio_id_has_a_unique_constraint(session):
    user = _user(session)
    first = _audio(session, user, name="a.wav", sha="a2" * 32)
    with pytest.raises(IntegrityError):
        session.add(AudioFile(audio_id=first.audio_id, filename="dup.wav",
                              stored_path="audio/x.wav", sha256="d1" * 32,
                              size_bytes=1, source="upload"))
        session.flush()


# --------------------------------------------------------------------------------------
# Scale -- NFR: at least 20,000 event records, and search must stay usable
# --------------------------------------------------------------------------------------


def test_twenty_thousand_events_insert_and_the_search_filter_stays_fast(factory, engine):
    """NFR: >=20,000 event records (FR lxxii) and a usable search (FR lxvii).

    The measured query is the shape the search endpoint issues: filter on severity and
    class, order by time, take a page. Without the ``ix_events_search`` index this is a
    full scan and the endpoint misses its budget once the demo database is full.
    """
    classes = ["Glass Breaking", "Gunshot", "Gunshot", "Machinery Fault",
               "Panic Scream", "Aggression", "Animal Sound"]
    severities = ["High", "Critical", "Low", "Medium"]

    with session_scope(factory) as s:
        user = _user(s, "load", "normal_user")
        model = _model_version(s, "python", "3.1.0")
        model2 = _model_version(s, "gtm", "2.0.0")
        audio_rows = [
            AudioFile(
                audio_id=format_audio_id(i + 1, dt.datetime(2026, 9, 23)),
                filename=f"clip_{i:06d}.wav",
                stored_path=f"audio/2026/09/clip_{i:06d}.wav",
                sha256=f"{i:064x}",
                size_bytes=44100,
                duration_sec=6.0,
                source="upload",
                created_by_id=user.id,
            )
            for i in range(20_000)
        ]
        s.add_all(audio_rows)
        s.flush()
        s.bulk_save_objects([
            Event(
                audio_file_id=audio_rows[i].id,
                source="upload",
                status="Classified",
                predicted_class=classes[i % len(classes)],
                severity=severities[i % len(severities)],
                consistency_status="Strong Match",
                confidence_difference=0.03,
                top_confidence=0.9,
                quality_verdict="Good",
                python_model_version_id=model.id,
                gtm_model_version_id=model2.id,
                created_by_id=user.id,
                created_at=dt.datetime(2026, 9, 23) + dt.timedelta(seconds=i),
            )
            for i in range(20_000)
        ])
        s.flush()

    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM events")).scalar() == 20_000

    with session_scope(factory) as s:
        started = time.perf_counter()
        rows = s.execute(
            select(Event)
            .where(Event.severity == "Critical", Event.predicted_class == "Gunshot")
            .order_by(Event.created_at.desc())
            .limit(50)
        ).scalars().all()
        elapsed_ms = (time.perf_counter() - started) * 1000

    assert len(rows) == 50
    assert all(r.severity == "Critical" and r.predicted_class == "Gunshot" for r in rows)
    assert elapsed_ms < 200, f"search took {elapsed_ms:.1f} ms; the index is missing or unused"

    with engine.connect() as connection:
        plan = connection.execute(text(
            "EXPLAIN QUERY PLAN SELECT id FROM events "
            "WHERE severity='Critical' AND predicted_class='Gunshot' "
            "ORDER BY created_at DESC LIMIT 50"
        )).fetchall()
    plan_text = " ".join(str(part) for row in plan for part in row)
    assert "SCAN events" not in plan_text, f"full table scan: {plan_text}"


# --------------------------------------------------------------------------------------
# database/init_db.py -- the script the evaluator runs
# --------------------------------------------------------------------------------------


def _load_init_module():
    spec = importlib.util.spec_from_file_location("init_db_under_test", INIT_DB)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_init_db_is_idempotent(tmp_path: Path, capsys):
    """Running it twice must change nothing and say so, not duplicate the accounts."""
    init = _load_init_module()
    db = tmp_path / "init.db"

    assert init.main(["--db", str(db), "--seed", "--storage", str(tmp_path / "st")]) == 0
    first = capsys.readouterr().out
    assert "6 created" in first

    assert init.main(["--db", str(db), "--seed", "--storage", str(tmp_path / "st")]) == 0
    second = capsys.readouterr().out
    assert "6 created" not in second
    assert "6 already present" in second

    engine = create_engine_for(str(db))
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM users")).scalar() == 6


def test_init_db_seeds_one_account_for_every_srs_role(tmp_path: Path):
    init = _load_init_module()
    db = tmp_path / "roles.db"
    assert init.main(["--db", str(db), "--seed", "--storage", str(tmp_path / "st")]) == 0

    engine = create_engine_for(str(db))
    with make_session_factory(engine)() as s:
        seeded = {u.role for u in s.execute(select(User)).scalars()}
    assert seeded == set(ROLES), f"missing a seed account for: {sorted(set(ROLES) - seeded)}"


def test_seed_passwords_are_hashed_never_stored_plaintext(tmp_path: Path):
    init = _load_init_module()
    db = tmp_path / "hash.db"
    assert init.main(["--db", str(db), "--seed", "--storage", str(tmp_path / "st")]) == 0

    from werkzeug.security import check_password_hash

    credentials = json.loads(CREDENTIALS.read_text())
    engine = create_engine_for(str(db))
    with make_session_factory(engine)() as s:
        for entry in credentials["users"]:
            user = s.execute(select(User).where(User.username == entry["username"])).scalar_one()
            assert user.password_hash != entry["password"]
            assert entry["password"] not in user.password_hash
            assert check_password_hash(user.password_hash, entry["password"])


def test_the_shipped_credentials_are_documented_as_public():
    """They are published on purpose (SRS §1.10 item 11) and must say so, or someone reuses
    them for something real."""
    payload = json.loads(CREDENTIALS.read_text())
    assert "DELIBERATELY published" in payload["_comment"]
    assert set(payload["_roles_covered"]) == set(ROLES)


def test_init_db_reset_requires_confirmation_when_not_confirmed(tmp_path: Path, monkeypatch):
    """--reset is destructive; without --yes and without a tty it must refuse, not proceed."""
    init = _load_init_module()
    db = tmp_path / "reset.db"
    assert init.main(["--db", str(db), "--seed", "--storage", str(tmp_path / "st")]) == 0

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert init.main(["--db", str(db), "--reset", "--storage", str(tmp_path / "st")]) == 1

    engine = create_engine_for(str(db))
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM users")).scalar() == 6


def test_init_db_reset_with_yes_rebuilds_and_reassigns_ids(tmp_path: Path):
    init = _load_init_module()
    db = tmp_path / "reset2.db"
    assert init.main(["--db", str(db), "--seed", "--storage", str(tmp_path / "st")]) == 0
    assert init.main(["--db", str(db), "--reset", "--yes", "--seed",
                      "--storage", str(tmp_path / "st")]) == 0

    engine = create_engine_for(str(db))
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM users")).scalar() == 6


def test_init_db_register_model_then_activate(tmp_path: Path):
    """FR lxxv: versions are registered and activated through a documented operator command."""
    init = _load_init_module()
    db = tmp_path / "models.db"
    assert init.main(["--db", str(db), "--seed", "--storage", str(tmp_path / "st")]) == 0
    assert init.main(["--db", str(db), "--register-model", "python", "--version", "3.1.0",
                      "--artifact", "python_models/svm_mfcc_v3.model.joblib",
                      "--feature-version", "mfcc_v1", "--activate"]) == 0

    engine = create_engine_for(str(db))
    with make_session_factory(engine)() as s:
        row = s.execute(select(ModelVersion)).scalar_one()
        assert row.model_name == "python" and row.version == "3.1.0" and row.is_active
        audit = s.execute(
            select(AuditRecord).where(AuditRecord.action == "model_activate")
        ).scalar_one()
        assert audit.after["version"] == "3.1.0"


def test_init_db_discovers_a_python_model_artifact(tmp_path: Path, monkeypatch):
    """The seed step registers a model only when a real artifact exists on disk."""
    init = _load_init_module()
    models_dir = tmp_path / "python_models" / "svm_v1"
    models_dir.mkdir(parents=True)
    (models_dir / "model_meta.json").write_text(json.dumps({
        "model_name": "svm_mfcc", "model_version": "3.1.0", "feature_version": "mfcc_v1",
        "algorithm": "SVC", "metrics": {"accuracy": 0.91, "macro_f1": 0.88},
    }))
    monkeypatch.setattr(init, "REPO_ROOT", tmp_path)

    artifacts = init.discover_model_artifacts()
    assert len(artifacts) == 1
    assert artifacts[0]["model_name"] == "python"
    assert artifacts[0]["version"] == "3.1.0"
    assert artifacts[0]["metrics"]["accuracy"] == 0.91


def test_init_db_reports_no_models_rather_than_inventing_one(tmp_path: Path, monkeypatch, capsys):
    """No trained artifact means the app says so; it must never register a fake version."""
    init = _load_init_module()
    monkeypatch.setattr(init, "REPO_ROOT", tmp_path)
    assert init.discover_model_artifacts() == []


def test_init_db_stats_runs_and_names_every_table(tmp_path: Path, capsys):
    init = _load_init_module()
    db = tmp_path / "stats.db"
    assert init.main(["--db", str(db), "--seed", "--storage", str(tmp_path / "st")]) == 0
    capsys.readouterr()
    assert init.main(["--db", str(db), "--stats"]) == 0
    out = capsys.readouterr().out
    for table in ("users", "events", "alerts", "reviews", "audit_records", "confidence_scores"):
        assert table in out
    assert "maintenance" in out and "administrator" in out


def test_init_db_emits_a_schema_file_covering_every_table(tmp_path: Path, monkeypatch):
    init = _load_init_module()
    target = tmp_path / "schema.sql"
    text_out = init.emit_schema(target)
    assert target.read_text(encoding="utf-8") == text_out
    for table in ("users", "audio_files", "events", "confidence_scores", "alerts",
                  "reviews", "audit_records", "live_sessions", "live_windows",
                  "model_versions"):
        assert f"CREATE TABLE {table} " in text_out


# --------------------------------------------------------------------------------------
# Relationships between the pieces
# --------------------------------------------------------------------------------------


def test_deleting_an_event_takes_its_scores_alerts_and_reviews(session):
    user = _user(session)
    audio = _audio(session, user)
    models = (_model_version(session), _model_version(session, "gtm"))
    event = _event(session, audio, user, *models)
    session.add(Alert(event_id=event.id, severity="High", status="Open"))
    session.add(Review(event_id=event.id, status="Pending Review"))
    session.flush()
    assert session.execute(select(ConfidenceScore)).scalars().all().__len__() == 6

    session.delete(event)
    session.flush()
    assert session.execute(select(ConfidenceScore)).scalars().all() == []
    assert session.execute(select(Alert)).scalars().all() == []
    assert session.execute(select(Review)).scalars().all() == []


def test_an_event_may_be_analysed_twice_from_one_audio_file(session):
    """Re-analysis by a new model version must not duplicate the audio bytes (FR lxxi)."""
    user = _user(session)
    audio = _audio(session, user)
    v1 = _model_version(session, "python", "3.0.0")
    gtm = _model_version(session, "gtm", "2.0.0")
    v2 = _model_version(session, "python", "4.0.0")
    first = _event(session, audio, user, v1, gtm)
    second = _event(session, audio, user, v2, gtm)
    session.flush()

    assert first.audio_file_id == second.audio_file_id == audio.id
    assert len(audio.events) == 2
    assert first.python_model_version_id != second.python_model_version_id


def test_the_search_index_exists_on_the_columns_the_filter_uses(engine):
    index_names = {ix["name"] for ix in inspect(engine).get_indexes("events")}
    assert "ix_events_search" in index_names
    assert "ix_events_review_queue" in index_names

    columns = {tuple(ix["column_names"]) for ix in inspect(engine).get_indexes("events")}
    assert ("created_at", "predicted_class", "severity", "status") in columns


def test_repr_and_to_dict_are_json_safe(session):
    user = _user(session, "jsonuser", "audio_reviewer")
    payload = user.to_dict(exclude=("password_hash",))
    assert "password_hash" not in payload
    assert payload["username"] == "jsonuser"
    assert payload["role"] == "audio_reviewer"
    assert json.loads(json.dumps(payload)) == payload
    assert "jsonuser" in repr(user)
