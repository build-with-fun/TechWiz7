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


def test_client_ip_ignores_forwarded_header_unless_a_proxy_is_trusted(tmp_path):
    """A client-supplied X-Forwarded-For must not change the address used for per-IP
    login limits; behind one trusted proxy, the entry that proxy appended is used."""
    from src.app import create_app
    from src.auth import client_ip

    app = create_app(TESTING=True, SST_DB_PATH=str(tmp_path / "a.db"),
                     SST_STORAGE_DIR=str(tmp_path / "s"), SST_LOAD_MODELS=False)
    headers = {"X-Forwarded-For": "6.6.6.6, 10.0.0.9"}
    with app.test_request_context("/", headers=headers, environ_base={"REMOTE_ADDR": "10.0.0.1"}):
        assert client_ip() == "10.0.0.1"
    app.config["SST_TRUSTED_PROXY_HOPS"] = 1
    with app.test_request_context("/", headers=headers, environ_base={"REMOTE_ADDR": "10.0.0.1"}):
        assert client_ip() == "10.0.0.9"
