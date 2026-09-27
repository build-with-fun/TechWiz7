"""The alerts and review pages render rows loaded from the database.

The pages expunge their rows and the templates then read alert.event / review.event. If
that wasn't loaded first, SQLAlchemy raises DetachedInstanceError (a 500). These tests
read the rows through a fresh session, like a real request.
"""

from sqlalchemy import select

from src.app import create_app
from src.auth import hash_password
from src.models import Alert, AudioFile, Event, Review, User


def _seed_app(tmp_path):
    """App with one critical event, alert and review."""
    app = create_app(TESTING=True, SST_DB_PATH=str(tmp_path / "app.db"),
                     SST_STORAGE_DIR=str(tmp_path / "storage"), SST_LOAD_MODELS=False)
    factory = app.config["SST_SESSION_FACTORY"]
    with factory() as session:
        admin = User(username="admin", email="admin@example.test", display_name="Admin",
                     password_hash=hash_password("IntegrationPass123!"), role="administrator")
        audio = AudioFile(audio_id="SST-2026-01-01-000001", filename="shot.wav",
                          stored_path="uploads/shot.wav", sha256="c" * 64,
                          size_bytes=32, source="upload")
        session.add_all([admin, audio])
        session.flush()
        event = Event(audio_file_id=audio.id, source="upload", status="Classified",
                      severity="Critical", predicted_class="Gunshot", final_class="Gunshot")
        session.add(event)
        session.flush()
        session.add_all([
            Alert(event_id=event.id, severity="Critical", status="Open",
                  message="Gunshot detected", recommended_action="Dispatch"),
            Review(event_id=event.id, status="Pending Review", priority=1,
                   reason_text="Models disagreed"),
        ])
        session.commit()
        admin_id = admin.id
    return app, factory, admin_id


def _signed_in_client(app, user_id):
    client = app.test_client()
    with app.test_request_context("/"):
        with client.session_transaction() as cookie:
            cookie["_user_id"] = str(user_id)
            cookie["_fresh"] = True
            cookie["_id"] = app.login_manager._session_identifier_generator()
    return client


def test_alerts_page_renders_an_alert_whose_event_was_never_loaded(tmp_path):
    app, factory, admin_id = _seed_app(tmp_path)
    # Nothing cached on the row.
    with factory() as session:
        session.execute(select(Alert)).scalars().all()
        session.expunge_all()

    response = _signed_in_client(app, admin_id).get("/alerts")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Gunshot detected" in body
    # The template reads alert.event.predicted_class.
    assert "Gunshot" in body


def test_reviews_page_renders_a_review_whose_event_was_never_loaded(tmp_path):
    app, factory, admin_id = _seed_app(tmp_path)
    with factory() as session:
        session.execute(select(Review)).scalars().all()
        session.expunge_all()

    response = _signed_in_client(app, admin_id).get("/reviews")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Models disagreed" in body
    assert "Gunshot" in body
