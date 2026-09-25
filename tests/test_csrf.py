"""Browser session writes require the token rendered by the same session."""

import re

from src.app import create_app


def test_login_rejects_missing_csrf_token(tmp_path):
    app = create_app(TESTING=True, SST_CSRF_ENABLED=True,
                     SST_DB_PATH=str(tmp_path / "csrf.db"),
                     SST_STORAGE_DIR=str(tmp_path / "storage"), SST_LOAD_MODELS=False)
    client = app.test_client()
    page = client.get("/login")
    assert page.status_code == 200
    token = re.search(rb'<meta name="csrf-token" content="([^"]+)"', page.data).group(1).decode()

    blocked = client.post("/api/auth/session", json={"username": "nobody", "password": "wrong"})
    assert blocked.status_code == 403

    checked = client.post("/api/auth/session", json={"username": "nobody", "password": "wrong"},
                          headers={"X-CSRF-Token": token})
    assert checked.status_code != 403
