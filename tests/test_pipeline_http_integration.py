"""Exercise the real analysis orchestrator across the HTTP and database boundary."""

import io
import base64
import wave
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select

from audio_preprocessing.pipeline import AudioPipeline
from src.app import create_app
from src.auth import hash_password
from src.inference.contract import PredictionResult
from src.models import AudioFile, Event, LiveWindow, User
from src.services.pipeline import AnalysisPipeline, PipelineModels, set_pipeline


class FixedPredictor:
    """Keep inference deterministic while testing the surrounding real workflow."""

    def __init__(self, name, classes):
        self.name = name
        self.classes = classes

    def predict_from_preprocessed(self, audio, *, origin):
        scores = {name: 0.01 for name in self.classes}
        scores["Background Noise"] = 0.91
        return PredictionResult(
            model_name=self.name,
            model_version="integration-test",
            predicted_class="Background Noise",
            confidence=0.91,
            confidences=scores,
            source_origin=origin,
        )


def _wav(seed=19) -> bytes:
    rng = np.random.default_rng(seed)
    samples = (rng.normal(0, 0.06, 32_000) * 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(samples.tobytes())
    return output.getvalue()


def test_upload_with_real_pipeline_writes_event_and_original(tmp_path):
    app = create_app(TESTING=True, SST_DB_PATH=str(tmp_path / "app.db"),
                     SST_STORAGE_DIR=str(tmp_path / "storage"), SST_LOAD_MODELS=False)
    classes = app.config["SST_CONFIG_STORE"].class_names()
    pipeline = AnalysisPipeline(PipelineModels(
        python=FixedPredictor("python", classes),
        gtm=FixedPredictor("gtm", classes),
        preprocessor=AudioPipeline(),
        feature_extractor=None,
    ), store=app.config["SST_CONFIG_STORE"])
    set_pipeline(pipeline)
    app.config["SST_PIPELINE"] = pipeline

    factory = app.config["SST_SESSION_FACTORY"]
    with factory() as session:
        user = User(username="operator", email="operator@example.test", display_name="Operator",
                    password_hash=hash_password("IntegrationPass123!"), role="normal_user")
        session.add(user)
        admin = User(username="admin", email="admin@example.test", display_name="Admin",
                     password_hash=hash_password("IntegrationPass123!"), role="administrator")
        session.add(admin)
        session.commit()
        user_id = user.id
        admin_id = admin.id

    client = app.test_client()
    with app.test_request_context("/"):
        with client.session_transaction() as cookie:
            cookie["_user_id"] = str(user_id)
            cookie["_fresh"] = True
            cookie["_id"] = app.login_manager._session_identifier_generator()

    response = client.post("/api/audio/upload", data={"file": (io.BytesIO(_wav()), "noise.wav")},
                           content_type="multipart/form-data")
    assert response.status_code == 201, response.get_json()
    event_id = response.get_json()["data"]["id"]
    assert event_id is not None
    with factory() as session:
        event = session.execute(select(Event).where(Event.id == event_id)).scalar_one()
        audio = session.execute(select(AudioFile).where(AudioFile.id == event.audio_file_id)).scalar_one()
        assert event.predicted_class == "Background Noise"
        assert app.config["SST_STORAGE"].resolve(audio.stored_path).read_bytes() == _wav()

    started = client.post("/api/live/sessions", json={"consent_ack": True})
    assert started.status_code == 201
    session_id = started.get_json()["data"]["id"]
    live = client.post(f"/api/live/sessions/{session_id}/windows", json={
        "seq": 1,
        "audio_b64": base64.b64encode(_wav(20)).decode(),
        "sample_rate": 16_000,
        "duration_sec": 2.0,
    })
    assert live.status_code == 200, live.get_json()
    live_event_id = live.get_json()["data"]["event_id"]
    with factory() as session:
        window = session.execute(select(LiveWindow).where(LiveWindow.session_id == session_id)).scalar_one()
        assert live_event_id == window.event_id
        assert session.get(Event, live_event_id) is not None

    forbidden_export = client.get("/api/export/events.csv")
    assert forbidden_export.status_code == 403

    with app.test_request_context("/"):
        with client.session_transaction() as cookie:
            cookie["_user_id"] = str(admin_id)
            cookie["_fresh"] = True
            cookie["_id"] = app.login_manager._session_identifier_generator()
    exported = client.get("/api/export/events.csv")
    assert exported.status_code == 200
    assert "Background Noise" in exported.get_data(as_text=True)
    workbook = client.get("/api/export/events.xlsx")
    assert workbook.status_code == 200
    assert workbook.data.startswith(b"PK")
    period = client.get("/api/reports/period?from=2020-01-01&to=2030-01-01")
    assert period.status_code == 200
    assert period.get_json()["data"]["totals"]["events"] == 2


def test_exported_models_complete_an_http_upload(tmp_path):
    """Catch model-loader and feature-contract failures hidden by fixed predictors."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "gtm_model/gtm_model.h5").exists():
        pytest.skip("The independently trained GTM export is not installed")
    if not (root / "python_models/best/model.joblib").exists():
        pytest.skip("The Python model artifact is not installed")

    app = create_app(TESTING=True, SST_DB_PATH=str(tmp_path / "app.db"),
                     SST_STORAGE_DIR=str(tmp_path / "storage"))
    assert app.config["SST_PIPELINE"] is not None
    assert app.test_client().get("/api/health/ready").status_code == 200
    factory = app.config["SST_SESSION_FACTORY"]
    with factory() as session:
        user = User(username="real_model", email="real@example.test",
                    password_hash=hash_password("IntegrationPass123!"), role="normal_user")
        session.add(user)
        session.commit()
        user_id = user.id

    client = app.test_client()
    with app.test_request_context("/"):
        with client.session_transaction() as cookie:
            cookie["_user_id"] = str(user_id)
            cookie["_fresh"] = True
            cookie["_id"] = app.login_manager._session_identifier_generator()

    response = client.post("/api/audio/upload", data={"file": (io.BytesIO(_wav()), "noise.wav")},
                           content_type="multipart/form-data")
    assert response.status_code == 201, response.get_json()
    event_id = response.get_json()["data"]["id"]
    with factory() as session:
        event = session.get(Event, event_id)
        assert event is not None
        assert {score.model_name for score in event.confidence_scores} == {"python", "gtm"}
        assert session.get(AudioFile, event.audio_file_id) is not None
