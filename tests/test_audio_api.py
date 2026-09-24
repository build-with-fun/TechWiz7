"""Tests for ``/api/audio`` and ``/api/events``.

Owner: sara.

The pipeline is stubbed, not the endpoints. A real trained model does not exist on disk
yet, and these tests are about the contract the console relies on -- the status codes, the
scoping rule, the duplicate gate, the shape of the response -- none of which depend on
which class the model picked. The stub returns a record shaped exactly like
``AnalysisPipeline.analyse`` so the serializer is exercised against the real thing.
"""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import pytest
from sqlalchemy import select

from src.app import create_app
from src.auth import hash_password, sign_in
from src.db import create_engine_for, create_schema, make_session_factory
from src.models import AudioFile, Event, User


# --------------------------------------------------------------------------------------
# A pipeline stub that returns a realistic record
# --------------------------------------------------------------------------------------

def _record(*, event_id: int = 1, sha256: str = "a" * 64, agree: bool = True,
            severity: str = "High", alert: bool = True, review: bool = False,
            verdict: str = "Good") -> dict:
    py = {"Gunshot": 0.93, "Glass Breaking": 0.58}
    gtm = {"Gunshot": 0.89, "Glass Breaking": 0.41}
    return {
        "ok": True,
        "status": "analysed",
        "origin": "upload",
        "created_at": "2026-09-24T13:00:00Z",
        "budget_sec": 8.0,
        "meta": {"filename": "probe.wav", "location": "Warehouse North"},
        "audio": {
            "audio_id": "SST-260924-000001",
            "duration_sec": 2.0,
            "sample_rate": 16000,
            "sha256": sha256,
            "fingerprint": f"fp-{sha256[:8]}",
            "fingerprint_version": "v1",
        },
        "quality": {"verdict": verdict, "problems": [], "summary": "clean"},
        "predictions": {
            "python": {"model_name": "python", "model_version": "svm_v1",
                       "predicted_class": "Gunshot", "confidence": 0.93, "confidences": py},
            "gtm": {"model_name": "gtm", "model_version": "tm_v1",
                    "predicted_class": "Gunshot" if agree else "Glass Breaking",
                    "confidence": 0.89, "confidences": gtm},
        },
        "comparison": {
            "consistency_status": "Strong Match" if agree else "Model Disagreement",
            "confidence_difference": 0.04,
        },
        "severity": {"severity": severity, "severity_display": severity,
                     "critical_class": severity == "Critical"},
        "alert": {"raised": alert},
        "review": {"required": review, "reason_text": "models disagreed" if review else None,
                   "priority": "critical" if review else None},
        "model_versions": {"python": {"version": "svm_v1"}, "gtm": {"version": "tm_v1"}},
        "decision": {"final_class": "Gunshot", "final_decision": "Likely Valid",
                     "severity_display": severity},
        "stored": True,
    }


class _StubPipeline:
    """Returns a canned record and records what it was asked to analyse."""

    def __init__(self):
        self.calls = []
        self.next_status = "analysed"
        self.next_quality_verdict = "Good"

    def analyse_bytes(self, raw, *, filename, origin, meta, persist=None):
        self.calls.append({"filename": filename, "origin": origin, "meta": meta,
                           "bytes": len(raw), "persist": persist is not None})
        if self.next_status == "rejected":
            return {**_record(), "status": "rejected",
                    "rejection": {"message": "too short to analyse"},
                    "quality": {"verdict": "Unusable", "problems": ["too_short"],
                                "summary": "0.2s is below the 0.5s minimum"}}
        return _record(verdict=self.next_quality_verdict)


@pytest.fixture()
def stub(monkeypatch):
    """Replace the process pipeline with a stub for the duration of the app."""
    import src.services.pipeline as pipeline_mod

    stub_ = _StubPipeline()
    monkeypatch.setattr(pipeline_mod, "get_pipeline", lambda: stub_)
    return stub_


# --------------------------------------------------------------------------------------
# App + users
# --------------------------------------------------------------------------------------

@pytest.fixture()
def app(tmp_path, stub):
    app_ = create_app(
        TESTING=True,
        SST_DB_PATH=str(tmp_path / "api.db"),
        SST_STORAGE_DIR=str(tmp_path / "storage"),
        SST_LOAD_MODELS=False,
    )
    return app_


@pytest.fixture()
def factory(app):
    return app.config["SST_SESSION_FACTORY"]


def _make_user(factory, *, username="viewer", role="normal_user"):
    with factory() as session:
        user = User(username=username, email=f"{username}@example.com",
                    display_name=username, password_hash=hash_password("Sup3rSecret!"),
                    role=role)
        session.add(user)
        session.commit()
        session.refresh(user)
        session.expunge(user)
        return user


@pytest.fixture()
def viewer(factory):
    return _make_user(factory)


@pytest.fixture()
def reviewer(factory):
    return _make_user(factory, username="rev", role="audio_reviewer")


def _login(app, user):
    """Sign in inside a request context so the session cookie is set."""
    with app.test_request_context("/"):
        sign_in(user)
    return user


def _client(app, user):
    """A test client carrying that user's signed session.

    The session is established inside a real request context on the app so the signed cookie
    the client carries is the one a browser would have received.
    """
    client = app.test_client()
    with app.test_request_context("/"):
        sign_in(user)
        from flask import session as flask_session

        cookie = flask_session.cookies.get("session", "")
    if cookie:
        client.set_cookie("localhost", "session", cookie)
    return client


def _upload(client, *, data=b"RIFF****", filename="probe.wav", **form):
    return client.post("/api/audio/upload", data={
        "file": (io.BytesIO(data), filename), **form
    }, content_type="multipart/form-data")


# --------------------------------------------------------------------------------------
# /api/audio/upload
# --------------------------------------------------------------------------------------

def test_upload_returns_201_with_the_event_shape(app, viewer, stub):
    client = _client(app, viewer)
    resp = _upload(client, location="Warehouse North")

    assert resp.status_code == 201
    body = resp.get_json()["data"]
    assert body["predicted_class"] == "Gunshot"
    assert body["severity"] == "High"
    assert body["consistency_status"] == "Strong Match"
    assert body["models"]["python"]["version"] == "svm_v1"
    assert body["models"]["gtm"]["version"] == "tm_v1"
    # FR lxxix: the pipeline was given the metadata, not just the bytes.
    assert stub.calls[0]["meta"]["location"] == "Warehouse North"


def test_upload_writes_the_event_and_audio_rows(app, factory, viewer):
    _upload(_client(app, viewer))
    with factory() as session:
        event = session.execute(select(Event)).scalar_one()
        assert event.predicted_class == "Gunshot"
        assert event.audio_file.sha256 == "a" * 64


def test_upload_of_the_same_bytes_is_a_409_naming_the_first(app, viewer):
    client = _client(app, viewer)
    first = _upload(client)
    assert first.status_code == 201
    again = _upload(client, filename="renamed.wav")

    assert again.status_code == 409
    assert again.get_json()["error"]["code"] == "duplicate_audio"
    assert again.get_json()["error"]["audio_id"].startswith("SST-")


def test_allow_duplicate_stores_a_second_event(app, viewer):
    client = _client(app, viewer)
    _upload(client)
    again = _upload(client, allow_duplicate="true")

    assert again.status_code == 201
    with app.config["SST_SESSION_FACTORY"]() as session:
        assert session.execute(select(Event)).scalars().all().__len__() == 2


def test_a_microphone_upload_needs_explicit_consent(app, viewer):
    resp = _upload(_client(app, viewer), source="microphone")
    assert resp.status_code == 422
    assert "consent" in resp.get_json()["error"]["message"].lower()


def test_a_microphone_upload_with_consent_is_accepted(app, viewer, stub):
    resp = _upload(_client(app, viewer), source="microphone", consent_ack="true")
    assert resp.status_code == 201
    # FR lxxix: the origin is mapped to the storage label the schema accepts.
    assert stub.calls[0]["origin"] == "live"


def test_unusable_audio_is_a_422_naming_the_reason(app, viewer, stub):
    stub.next_status = "rejected"
    resp = _upload(_client(app, viewer))

    assert resp.status_code == 422
    err = resp.get_json()["error"]
    assert err["code"] == "quality_unusable"
    assert "below the 0.5s minimum" in err["message"]


def test_an_unsupported_type_is_refused_before_decoding(app, viewer):
    resp = _upload(_client(app, viewer), filename="notes.txt")
    assert resp.status_code == 415
    assert resp.get_json()["error"]["code"] == "unsupported_media_type"


def test_a_missing_file_part_is_a_422(app, viewer):
    resp = _client(app, viewer).post("/api/audio/upload", data={})
    assert resp.status_code == 422


def test_an_anonymous_upload_is_not_allowed(app):
    resp = app.test_client().post("/api/audio/upload", data={
        "file": (io.BytesIO(b"x"), "probe.wav")
    }, content_type="multipart/form-data")
    assert resp.status_code in (401, 403)


# --------------------------------------------------------------------------------------
# /api/events -- scoping
# --------------------------------------------------------------------------------------

def _seed_event(factory, user, *, predicted="Gunshot", severity="High"):
    from src.services.persistence import EventStore
    from src.db import StorageLayout

    layout = StorageLayout(Path("/tmp") / f"sst-test-{user.id}")
    layout.ensure()
    record = _record(severity=severity)
    record["predictions"]["python"]["predicted_class"] = predicted
    record["decision"]["final_class"] = predicted
    with factory() as session:
        result = EventStore(session, storage=layout, actor=user).store(record)
    return result["event_ids"][0]


def test_a_viewer_sees_only_their_own_events(app, factory, viewer, reviewer):
    mine = _seed_event(factory, viewer)
    _seed_event(factory, reviewer, predicted="Glass Breaking")

    resp = _client(app, viewer).get("/api/events")
    assert resp.status_code == 200
    page = resp.get_json()
    assert page["meta"]["total"] == 1
    assert page["data"][0]["id"] == mine


def test_a_viewer_asking_for_someone_elses_event_gets_a_404(app, factory, viewer, reviewer):
    theirs = _seed_event(factory, reviewer)
    resp = _client(app, viewer).get(f"/api/events/{theirs}")
    assert resp.status_code == 404


def test_a_reviewer_sees_every_event(app, factory, viewer, reviewer):
    _seed_event(factory, viewer)
    _seed_event(factory, reviewer, predicted="Glass Breaking")

    resp = _client(app, reviewer).get("/api/events")
    assert resp.get_json()["meta"]["total"] == 2


def test_unknown_event_id_is_a_404(app, viewer):
    resp = _client(app, viewer).get("/api/events/9999")
    assert resp.status_code == 404


# --------------------------------------------------------------------------------------
# /api/events -- filters and sort
# --------------------------------------------------------------------------------------

def test_severity_filter_narrows_the_list(app, factory, viewer):
    _seed_event(factory, viewer, severity="High")
    _seed_event(factory, viewer, severity="Low")

    resp = _client(app, viewer).get("/api/events?severity=High")
    page = resp.get_json()
    assert page["meta"]["total"] == 1
    assert page["data"][0]["severity"] == "High"


def test_an_unknown_severity_is_a_422_not_a_silent_match(app, viewer):
    resp = _client(app, viewer).get("/api/events?severity=Catastrophic")
    assert resp.status_code == 422
    assert resp.get_json()["error"]["code"] == "validation_error"


def test_an_unknown_sort_is_a_422(app, viewer):
    resp = _client(app, viewer).get("/api/events?sort=random")
    assert resp.status_code == 422


def test_pagination_meta_is_present(app, factory, viewer):
    for _ in range(3):
        _seed_event(factory, viewer)

    resp = _client(app, viewer).get("/api/events?per_page=2")
    page = resp.get_json()
    assert page["meta"]["total"] == 3
    assert page["meta"]["per_page"] == 2
    assert page["meta"]["pages"] >= 2
    assert len(page["data"]) == 2


# --------------------------------------------------------------------------------------
# Evidence, audio, flag, delete
# --------------------------------------------------------------------------------------

def test_evidence_returns_both_models_distributions(app, factory, viewer):
    event_id = _seed_event(factory, viewer)
    resp = _client(app, viewer).get(f"/api/events/{event_id}/evidence")

    assert resp.status_code == 200
    models = resp.get_json()["data"]["models"]
    assert set(models) == {"python", "gtm"}
    assert models["python"]["predicted_class"] == "Gunshot"
    assert models["python"]["confidences"][0]["is_top"] is True


def test_flagging_an_event_is_audited_not_a_reclassification(app, factory, viewer):
    from src.models import AuditRecord

    event_id = _seed_event(factory, viewer)
    resp = _client(app, viewer).post(f"/api/events/{event_id}/flag",
                                     json={"note": "not sure about this one"})
    assert resp.status_code == 200

    with factory() as session:
        actions = [a.action for a in session.execute(select(AuditRecord)).scalars()]
        event = session.execute(select(Event).where(Event.id == event_id)).scalar_one()
    assert "event_flagged" in actions
    # The classification is untouched: flagging is a request for attention.
    assert event.predicted_class == "Gunshot"


def test_a_viewer_cannot_delete_an_event(app, factory, viewer):
    event_id = _seed_event(factory, viewer)
    resp = _client(app, viewer).delete(f"/api/events/{event_id}")
    assert resp.status_code in (403, 404)


def test_a_reviewer_can_delete_and_it_is_audited_with_a_before_image(app, factory, reviewer):
    from src.models import AuditRecord

    event_id = _seed_event(factory, reviewer)
    resp = _client(app, reviewer).delete(f"/api/events/{event_id}")
    assert resp.status_code == 200

    with factory() as session:
        assert session.execute(select(Event)).scalar_one_or_none() is None
        rows = [a for a in session.execute(select(AuditRecord)).scalars()
                if a.action == "event_deleted"]
    assert rows
    assert rows[0].before["sha256"] == "a" * 64
