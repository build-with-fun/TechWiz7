"""Audio evidence must actually download, and the alert history must serialise.

Two defects this pins down:

* ``/api/audio/<id>/download`` and ``/api/events/<id>/audio`` read
  ``audio.content_type``. No such column exists -- the format lives in the stored path
  suffix and in ``container_format``/``original_format`` -- so both endpoints raised
  ``AttributeError`` and returned 500 for *every* audio file. The download of the evidence
  is the whole point of keeping it.
* ``/api/alerts/history`` built its response with ``_alert_to_dict`` after the session had
  closed; ``alert.event`` was an unloaded relationship on a detached row, so the endpoint
  500'd once any alert had been resolved.
"""


from src.app import create_app
from src.auth import hash_password
from src.models import Alert, AudioFile, Event, User


def _backend_app(tmp_path):
    """An app over a real dataset: one classified event with bytes on disk and an alert."""
    storage = tmp_path / "storage"
    uploads = storage / "uploads"
    uploads.mkdir(parents=True)
    (uploads / "shot.wav").write_bytes(b"RIFF....WAVEfmt ")

    app = create_app(TESTING=True, SST_DB_PATH=str(tmp_path / "app.db"),
                     SST_STORAGE_DIR=str(storage), SST_LOAD_MODELS=False)
    factory = app.config["SST_SESSION_FACTORY"]
    with factory() as session:
        admin = User(username="admin", email="admin@example.test", display_name="Admin",
                     password_hash=hash_password("IntegrationPass123!"), role="administrator")
        audio = AudioFile(audio_id="SST-2026-01-01-000001", filename="shot.wav",
                          stored_path="uploads/shot.wav", container_format="wav",
                          sha256="d" * 64, size_bytes=16, source="upload",
                          duration_sec=1.0)
        session.add_all([admin, audio])
        session.flush()
        event = Event(audio_file_id=audio.id, source="upload", status="Classified",
                      severity="Critical", predicted_class="Gunshot", final_class="Gunshot")
        session.add(event)
        session.flush()
        session.add(Alert(event_id=event.id, severity="Critical", status="Closed",
                          message="Gunshot detected", recommended_action="Dispatch"))
        session.commit()
        admin_id, audio_id, event_id = admin.id, audio.id, event.id
    return app, factory, admin_id, audio_id, event_id


def _client(app, user_id):
    client = app.test_client()
    with app.test_request_context("/"):
        with client.session_transaction() as cookie:
            cookie["_user_id"] = str(user_id)
            cookie["_fresh"] = True
            cookie["_id"] = app.login_manager._session_identifier_generator()
    return client


def test_content_type_is_derived_from_the_stored_path():
    wav = AudioFile(stored_path="uploads/clip.wav")
    assert wav.content_type == "audio/wav"
    # A purged record keeps its recorded format, so the header is still right.
    purged = AudioFile(stored_path="", container_format="mp3")
    assert purged.content_type == "audio/mpeg"
    # Unknown and absent formats fall back rather than raising.
    assert AudioFile(stored_path="clip.zzz").content_type == "application/octet-stream"
    assert AudioFile(stored_path="").content_type == "application/octet-stream"


def test_every_accepted_audio_format_has_a_mime_type():
    from src.models import AUDIO_MIME_TYPES

    accepted = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    assert accepted <= set(AUDIO_MIME_TYPES)
    assert all(mime.startswith("audio/") for mime in AUDIO_MIME_TYPES.values())


def test_audio_download_serves_the_bytes_with_a_real_mime_type(tmp_path):
    app, _factory, admin_id, audio_id, _event_id = _backend_app(tmp_path)
    response = _client(app, admin_id).get(f"/api/audio/{audio_id}/download")

    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("audio/")
    assert response.get_data() == b"RIFF....WAVEfmt "


def test_event_audio_endpoint_serves_the_bytes(tmp_path):
    app, _factory, admin_id, _audio_id, event_id = _backend_app(tmp_path)
    response = _client(app, admin_id).get(f"/api/events/{event_id}/audio")

    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("audio/")
    assert response.get_data() == b"RIFF....WAVEfmt "


def test_alert_history_serialises_a_resolved_alert(tmp_path):
    app, factory, admin_id, _audio_id, event_id = _backend_app(tmp_path)
    # Nothing about the alert is cached on this instance, as in a real request.
    with factory() as session:
        session.expunge_all()

    response = _client(app, admin_id).get("/api/alerts/history")

    assert response.status_code == 200
    body = response.get_json()
    # ``outcome_counts`` is fed by a GROUP BY on severity, so it counts severities.
    assert body["meta"]["outcome_counts"] == {"Critical": 1}
    assert body["data"][0]["event_id"] == event_id
    assert body["data"][0]["message"] == "Gunshot detected"
