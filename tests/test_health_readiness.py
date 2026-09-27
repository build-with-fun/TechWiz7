"""Readiness reflects whether audio can actually be analysed."""

from src.app import create_app


def test_readiness_reports_missing_analysis_models(tmp_path):
    app = create_app(TESTING=True, SST_LOAD_MODELS=False,
                     SST_DB_PATH=str(tmp_path / "app.db"),
                     SST_STORAGE_DIR=str(tmp_path / "storage"))
    client = app.test_client()

    liveness = client.get("/api/health")
    assert liveness.status_code == 200
    assert liveness.get_json()["analysis_ready"] is False
    readiness = client.get("/api/health/ready")
    assert readiness.status_code == 503
    assert readiness.get_json()["reason"] == "analysis models unavailable"
