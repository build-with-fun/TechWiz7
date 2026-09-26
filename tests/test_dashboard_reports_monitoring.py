"""Dashboards (FR lxiii, lxv), analytics (lxviii), report (lxix), anomalies (lxxviii),
metadata (x) and the live-window payload (xxxiv), through the real Flask app.

Inference is deterministic here (FixedPredictor); these tests check what the product
does with a result, not how accurate the models are.
"""

import base64
import io
import wave

import numpy as np
import pytest
from sqlalchemy import select

from audio_preprocessing.pipeline import AudioPipeline
from src.app import create_app
from src.auth import hash_password
from src.db import record_audit, session_scope
from src.inference.contract import PredictionResult
from src.models import AudioFile, AuditRecord, User
from src.services.pipeline import AnalysisPipeline, PipelineModels, set_pipeline


class FixedPredictor:
    def __init__(self, name, classes, top="Gunshot", conf=0.9):
        self.name, self.classes, self.top, self.conf = name, classes, top, conf

    def predict_from_preprocessed(self, audio, *, origin):
        rest = (1.0 - self.conf) / (len(self.classes) - 1)
        scores = {c: rest for c in self.classes}
        scores[self.top] = self.conf
        return PredictionResult(model_name=self.name, model_version="test",
                                predicted_class=self.top, confidence=self.conf,
                                confidences=scores, source_origin=origin)


def wav_bytes(seed=3, seconds=2.0, rate=16000):
    rng = np.random.default_rng(seed)
    samples = (rng.normal(0, 0.08, int(rate * seconds)) * 32767).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return out.getvalue()


@pytest.fixture()
def app(tmp_path):
    app = create_app(TESTING=True, SST_DB_PATH=str(tmp_path / "app.db"),
                     SST_STORAGE_DIR=str(tmp_path / "storage"), SST_LOAD_MODELS=False)
    classes = app.config["SST_CONFIG_STORE"].class_names()
    pipeline = AnalysisPipeline(PipelineModels(
        python=FixedPredictor("python", classes),
        gtm=FixedPredictor("gtm", classes, top="Aggression", conf=0.6),
        preprocessor=AudioPipeline(), feature_extractor=None,
    ), store=app.config["SST_CONFIG_STORE"])
    set_pipeline(pipeline)
    app.config["SST_PIPELINE"] = pipeline
    with app.config["SST_SESSION_FACTORY"]() as session:
        for name, role in (("user1", "normal_user"), ("user2", "normal_user"),
                           ("admin", "administrator")):
            session.add(User(username=name, email=f"{name}@example.test", display_name=name,
                             password_hash=hash_password("TestPass123!x"), role=role))
        session.commit()
    yield app
    set_pipeline(None)


def login(app, client, username):
    with app.config["SST_SESSION_FACTORY"]() as session:
        uid = session.execute(select(User.id).where(User.username == username)).scalar_one()
    with app.test_request_context("/"):
        with client.session_transaction() as cookie:
            cookie["_user_id"] = str(uid)
            cookie["_fresh"] = True
            cookie["_id"] = app.login_manager._session_identifier_generator()


def upload(client, data, name="clip.wav"):
    return client.post("/api/audio/upload", data={"file": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_upload_stores_source_rate_and_bit_depth(app):
    client = app.test_client()
    login(app, client, "user1")
    assert upload(client, wav_bytes(rate=22050)).status_code == 201
    with app.config["SST_SESSION_FACTORY"]() as session:
        audio = session.execute(select(AudioFile)).scalar_one()
        assert audio.bit_depth == 16
        assert audio.sample_rate == 22050   # the file as received, not the 16 kHz working rate


def test_normal_user_dashboard_shows_only_their_own_uploads(app):
    client = app.test_client()
    login(app, client, "user1")
    assert upload(client, wav_bytes(1), "mine.wav").status_code == 201
    login(app, client, "user2")
    assert upload(client, wav_bytes(2), "theirs.wav").status_code == 201
    page = client.get("/dashboard")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "theirs.wav" in html and "mine.wav" not in html
    assert "Administrator overview" not in html


def test_admin_dashboard_has_srs_metrics_and_anomalies(app):
    client = app.test_client()
    login(app, client, "user1")
    upload(client, wav_bytes(4))
    with session_scope(app.config["SST_SESSION_FACTORY"]) as session:
        for _ in range(12):
            record_audit(session, action="login_failure", outcome="failure", detail="test")
    login(app, client, "admin")
    html = client.get("/dashboard").get_data(as_text=True)
    for text in ("Average top-class confidence", "Model disagreements",
                 "Poor-quality recordings", "Detection trend", "failed sign-ins"):
        assert text in html, text
    body = client.get("/api/admin/monitoring/anomalies").get_json()
    kinds = {a["kind"] for a in body["data"]["anomalies"]}
    assert "failed_logins" in kinds


def test_failed_upload_is_audited_for_anomaly_detection(app):
    client = app.test_client()
    login(app, client, "user1")
    silent = io.BytesIO()
    with wave.open(silent, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 32000)
    response = upload(client, silent.getvalue(), "silence.wav")
    assert response.status_code == 422
    with app.config["SST_SESSION_FACTORY"]() as session:
        rows = session.execute(select(AuditRecord).where(
            AuditRecord.action == "audio_upload", AuditRecord.outcome == "failure")).scalars().all()
    assert rows


def test_event_report_contains_every_fr_lxix_item(app):
    client = app.test_client()
    login(app, client, "user1")
    event_id = upload(client, wav_bytes(5)).get_json()["data"]["id"]
    login(app, client, "admin")
    response = client.get(f"/api/reports/event/{event_id}?format=html")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    for text in ("Audio metadata", "Bit depth", "Python prediction", "Teachable Machine prediction",
                 "Confidence scores, every class", "Top-class confidence difference",
                 "Audio quality", "Severity", "Alert status", "Review"):
        assert text in html, text
    assert html.count("data:image/png;base64,") == 2      # waveform + spectrogram
    assert "Machinery Fault" in html                       # all ten classes are listed


def test_analytics_page_reports_false_positive_and_alert_response_sections(app):
    client = app.test_client()
    login(app, client, "admin")
    html = client.get("/analytics").get_data(as_text=True)
    for text in ("Confidence distribution", "False positives and false negatives", "Alert response"):
        assert text in html


def test_live_window_payload_carries_consistency_and_top3(app):
    client = app.test_client()
    login(app, client, "user1")
    started = client.post("/api/live/sessions", json={"consent_ack": True})
    session_id = started.get_json()["data"]["id"]
    window = client.post(f"/api/live/sessions/{session_id}/windows", json={
        "seq": 1, "audio_b64": base64.b64encode(wav_bytes(6)).decode(),
        "sample_rate": 16000, "duration_sec": 2.0})
    assert window.status_code == 200, window.get_json()
    data = window.get_json()["data"]
    # The models disagree (Gunshot vs Aggression), so the status must say so. Before the
    # fix on 26 Sep this key was read from the wrong field and was always null.
    assert data["consistency_status"] == "Model Disagreement"
    assert [x["class"] for x in data["top3"]["python"]][0] == "Gunshot"
    assert len(data["top3"]["gtm"]) == 3
    assert data["review_required"] is True


def _signal_wav(kind: str, gain: float = 1.0) -> bytes:
    rate = 16000
    t = np.arange(rate * 3) / rate
    if kind == "sweep":
        y = np.sin(2 * np.pi * (300 + 900 * t) * t)
    else:
        y = np.sign(np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 1800 * t)
    rng = np.random.default_rng(0 if kind == "sweep" else 1)
    y = 0.5 * y + 0.02 * rng.normal(size=t.size)
    pcm = (np.clip(y * gain, -1, 1) * 32767).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return out.getvalue()


def test_two_stage_near_duplicate_flags_a_quieter_copy_but_not_a_different_sound(app):
    """FR lxxiv: a volume-adjusted copy has different bytes (no SHA-256 match) but must
    be recognised; an unrelated recording must not be."""
    client = app.test_client()
    login(app, client, "user1")
    first = upload(client, _signal_wav("sweep"), "original.wav")
    assert first.status_code == 201
    copy = upload(client, _signal_wav("sweep", gain=0.3), "quieter_copy.wav")
    assert copy.status_code == 201
    other = upload(client, _signal_wav("pulses"), "different.wav")
    assert other.status_code == 201
    with app.config["SST_SESSION_FACTORY"]() as session:
        rows = {a.filename: a for a in session.execute(select(AudioFile)).scalars()}
    assert rows["quieter_copy.wav"].near_duplicate_of_id == rows["original.wav"].id
    assert rows["different.wav"].near_duplicate_of_id is None


def test_an_acknowledged_alert_can_still_be_escalated_then_dismissed(app):
    """FR lv. Before 26 Sep any action on a non-Open alert returned 200 and changed nothing."""
    from src.models import Alert

    client = app.test_client()
    login(app, client, "user1")
    event_id = upload(client, wav_bytes(9)).get_json()["data"]["id"]
    with session_scope(app.config["SST_SESSION_FACTORY"]) as session:
        alert = Alert(event_id=event_id, severity="High", status="Open", rule_class="Gunshot")
        session.add(alert)
        session.flush()
        alert_id = alert.id
    login(app, client, "admin")
    assert client.post(f"/api/alerts/{alert_id}/acknowledge", json={}).get_json()["data"]["status"] == "Acknowledged"
    escalated = client.post(f"/api/alerts/{alert_id}/escalate", json={"severity": "Critical"})
    assert escalated.get_json()["data"]["status"] == "Escalated", escalated.get_json()
    dismissed = client.post(f"/api/alerts/{alert_id}/dismiss", json={"reason": "fireworks, checked on CCTV"})
    assert dismissed.get_json()["data"]["status"] == "Dismissed"
    again = client.post(f"/api/alerts/{alert_id}/acknowledge", json={})
    assert again.status_code == 409
    with app.config["SST_SESSION_FACTORY"]() as session:
        row = session.get(Alert, alert_id)
        assert row.acknowledged_by_id is not None and row.resolution_note == "fireworks, checked on CCTV"


def test_an_agreed_confident_critical_upload_raises_one_alert(tmp_path):
    """FR xlii/xlvi: uploads are confirmed by agreement, quality and confidence, not by
    waiting for windows that will never come; a live window still needs a streak."""
    app = create_app(TESTING=True, SST_DB_PATH=str(tmp_path / "a.db"),
                     SST_STORAGE_DIR=str(tmp_path / "s"), SST_LOAD_MODELS=False)
    classes = app.config["SST_CONFIG_STORE"].class_names()
    pipeline = AnalysisPipeline(PipelineModels(
        python=FixedPredictor("python", classes, top="Glass Breaking", conf=0.95),
        gtm=FixedPredictor("gtm", classes, top="Glass Breaking", conf=0.9),
        preprocessor=AudioPipeline(), feature_extractor=None,
    ), store=app.config["SST_CONFIG_STORE"])
    set_pipeline(pipeline)
    app.config["SST_PIPELINE"] = pipeline
    with app.config["SST_SESSION_FACTORY"]() as session:
        session.add(User(username="u", email="u@example.test", display_name="u",
                         password_hash=hash_password("TestPass123!x"), role="normal_user"))
        session.commit()
    client = app.test_client()
    login(app, client, "u")
    # An event over a quiet background, so the quality gate passes. (White noise is rightly
    # rated Poor; so is a perfectly steady tone, a known limit of the SNR estimate.)
    t = np.arange(32000) / 16000
    rng = np.random.default_rng(1)
    chirp = (np.where((t > 0.5) & (t < 1.2), 0.5 * np.sin(2 * np.pi * 900 * t), 0.0)
             + 0.005 * rng.normal(size=t.size)).astype("float32")
    first = pipeline.analyse_samples(chirp, 16000, origin="upload")
    assert first["quality"]["verdict"] in {"Good", "Acceptable"}
    assert first["alert"]["raised"] is True
    live = pipeline.analyse_samples(chirp * 0.9, 16000, origin="live")
    assert live["alert"]["needed"] >= 2        # live still waits for a streak
    set_pipeline(None)
