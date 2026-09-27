"""The event page renders the top score's latency.

Regression test: the template called a macro that didn't exist, but only when the top
score had a latency, so most events returned 500 while the tests passed.
"""

from src.app import create_app
from src.auth import hash_password
from src.models import AudioFile, ConfidenceScore, Event, User


def _seed_app(tmp_path, *, latency):
    """App with one event whose top score has a latency."""
    app = create_app(TESTING=True, SST_DB_PATH=str(tmp_path / "app.db"),
                     SST_STORAGE_DIR=str(tmp_path / "storage"), SST_LOAD_MODELS=False)
    factory = app.config["SST_SESSION_FACTORY"]
    with factory() as session:
        admin = User(username="admin", email="admin@example.test", display_name="Admin",
                     password_hash=hash_password("IntegrationPass123!"), role="administrator")
        audio = AudioFile(audio_id="SST-2026-01-01-000001", filename="shot.wav",
                          stored_path="uploads/shot.wav", sha256="e" * 64,
                          size_bytes=32, source="upload")
        session.add_all([admin, audio])
        session.flush()
        event = Event(audio_file_id=audio.id, source="upload", status="Classified",
                      severity="Critical", predicted_class="Gunshot", final_class="Gunshot")
        session.add(event)
        session.flush()
        for model in ("python", "gtm"):
            session.add_all([
                ConfidenceScore(event_id=event.id, model_name=model, class_name="Gunshot",
                                confidence=0.82, rank=1, is_top=True, latency_sec=latency),
                ConfidenceScore(event_id=event.id, model_name=model, class_name="Glass Breaking",
                                confidence=0.11, rank=2, is_top=False, latency_sec=latency),
            ])
        session.commit()
        ids = (admin.id, event.id)
    return app, ids


def _signed_in_client(app, user_id):
    client = app.test_client()
    with app.test_request_context("/"):
        with client.session_transaction() as cookie:
            cookie["_user_id"] = str(user_id)
            cookie["_fresh"] = True
            cookie["_id"] = app.login_manager._session_identifier_generator()
    return client


def test_event_detail_renders_the_top_score_latency(tmp_path):
    app, (admin_id, event_id) = _seed_app(tmp_path, latency=0.0123)

    response = _signed_in_client(app, admin_id).get(f"/events/{event_id}")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    # Whole milliseconds, like live.js ("Inference 12 ms").
    assert "Inference 12 ms" in body


def test_event_detail_renders_when_the_top_score_has_no_latency(tmp_path):
    # No latency: the line is skipped.
    app, (admin_id, event_id) = _seed_app(tmp_path, latency=None)

    response = _signed_in_client(app, admin_id).get(f"/events/{event_id}")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Gunshot" in body
    assert "Inference" not in body
