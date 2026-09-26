"""The event detail page must render the top-score latency.

The page detaches (``expunge``) every row it hands the template, and the template renders
"Inference ..." through the ``ms`` macro only when the model's *top* score carries a
``latency_sec``. When the macro the template calls does not exist, Jinja raises
``UndefinedError`` -> HTTP 500 -- but only on the events that actually reach that line.

The blind spot this closes: every other test either never rendered this page at all, or
built a prediction with no latency, so the branch was never entered and the missing macro
went unnoticed. On the real database that meant /events/1-3 rendered and /events/4-191
were 500s.
"""

from src.app import create_app
from src.auth import hash_password
from src.models import AudioFile, ConfidenceScore, Event, User


def _seed_app(tmp_path, *, latency):
    """An app holding one classified event whose top score carries ``latency`` seconds."""
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
    # The macro renders whole milliseconds, matching live.js ("Inference 12 ms").
    assert "Inference 12 ms" in body


def test_event_detail_renders_when_the_top_score_has_no_latency(tmp_path):
    # The branch the template skips: this shape is why the missing macro stayed hidden.
    app, (admin_id, event_id) = _seed_app(tmp_path, latency=None)

    response = _signed_in_client(app, admin_id).get(f"/events/{event_id}")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Gunshot" in body
    assert "Inference" not in body
