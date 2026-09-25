"""Retention preview and purge must describe and perform the same deletion."""

from datetime import timedelta

from sqlalchemy import select

from src.app import create_app
from src.auth import hash_password
from src.models import AudioFile, Event, User, utcnow


def test_retention_preview_then_purge_removes_expired_unheld_evidence(tmp_path):
    app = create_app(TESTING=True, SST_DB_PATH=str(tmp_path / "app.db"),
                     SST_STORAGE_DIR=str(tmp_path / "storage"), SST_LOAD_MODELS=False)
    factory = app.config["SST_SESSION_FACTORY"]
    storage = app.config["SST_STORAGE"]
    old = utcnow() - timedelta(days=2000)
    path = storage.root / "uploads" / "old.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"old evidence")
    with factory() as session:
        admin = User(username="admin", email="admin@example.test", display_name="Admin",
                     password_hash=hash_password("IntegrationPass123!"), role="administrator")
        audio = AudioFile(audio_id="SST-2020-01-01-000001", filename="old.wav",
                          stored_path="uploads/old.wav", sha256="a" * 64,
                          size_bytes=12, source="upload", created_at=old)
        session.add_all([admin, audio])
        session.flush()
        event = Event(audio_file_id=audio.id, source="upload", status="Classified",
                      severity="Low", created_at=old)
        session.add(event)
        session.commit()
        admin_id = admin.id

    client = app.test_client()
    with app.test_request_context("/"):
        with client.session_transaction() as cookie:
            cookie["_user_id"] = str(admin_id)
            cookie["_fresh"] = True
            cookie["_id"] = app.login_manager._session_identifier_generator()

    preview = client.post("/api/admin/retention/preview")
    assert preview.status_code == 200
    assert preview.json["data"]["plan"]["audio"] == 1
    assert preview.json["data"]["plan"]["event_records"] == 1
    assert path.exists()

    purged = client.post("/api/admin/retention/purge?dry_run=0")
    assert purged.status_code == 200
    assert purged.json["data"]["plan"] == preview.json["data"]["plan"]
    assert not path.exists()
    with factory() as session:
        assert session.execute(select(Event)).scalars().all() == []
        assert session.execute(select(AudioFile)).scalar_one().stored_path == ""


def test_retained_event_reports_expired_audio_without_server_error(tmp_path):
    app = create_app(TESTING=True, SST_DB_PATH=str(tmp_path / "app.db"),
                     SST_STORAGE_DIR=str(tmp_path / "storage"), SST_LOAD_MODELS=False)
    factory = app.config["SST_SESSION_FACTORY"]
    old = utcnow() - timedelta(days=400)
    path = app.config["SST_STORAGE"].root / "uploads" / "expired.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"expired evidence")
    with factory() as session:
        admin = User(username="admin", email="admin@example.test", display_name="Admin",
                     password_hash=hash_password("IntegrationPass123!"), role="administrator")
        audio = AudioFile(audio_id="SST-2020-01-01-000002", filename="expired.wav",
                          stored_path="uploads/expired.wav", sha256="b" * 64,
                          size_bytes=16, source="upload", created_at=old)
        session.add_all([admin, audio])
        session.flush()
        event = Event(audio_file_id=audio.id, source="upload", status="Classified",
                      severity="Low")
        session.add(event)
        session.commit()
        admin_id, event_id = admin.id, event.id

    client = app.test_client()
    with app.test_request_context("/"):
        with client.session_transaction() as cookie:
            cookie["_user_id"] = str(admin_id)
            cookie["_fresh"] = True
            cookie["_id"] = app.login_manager._session_identifier_generator()
    assert client.post("/api/admin/retention/purge?dry_run=0").status_code == 200
    assert not path.exists()
    with factory() as session:
        assert session.get(Event, event_id) is not None
    for url in (f"/api/events/{event_id}/audio", f"/api/events/{event_id}/visuals",
                f"/events/{event_id}/audio"):
        response = client.get(url)
        assert response.status_code == 410, (url, response.get_data(as_text=True)[:200])
