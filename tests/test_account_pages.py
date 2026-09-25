"""Registration and profile updates keep role assignment on the server."""

from sqlalchemy import select

from src.app import create_app
from src.models import User


def test_register_and_update_own_profile(tmp_path):
    app = create_app(TESTING=True, SST_DB_PATH=str(tmp_path / "app.db"),
                     SST_STORAGE_DIR=str(tmp_path / "storage"), SST_LOAD_MODELS=False)
    client = app.test_client()
    assert client.get("/register").status_code == 200
    response = client.post("/register", data={
        "username": "new_user", "email": "new@example.test", "display_name": "New User",
        "password": "NewUserPass123!", "role": "administrator",
    })
    assert response.status_code == 302
    with app.config["SST_SESSION_FACTORY"]() as session:
        user = session.execute(select(User).where(User.username == "new_user")).scalar_one()
        assert user.role == "normal_user"

    assert client.get("/profile").status_code == 200
    updated = client.post("/profile", data={
        "display_name": "A New User", "email": "renamed@example.test",
    })
    assert updated.status_code == 302
    with app.config["SST_SESSION_FACTORY"]() as session:
        user = session.execute(select(User).where(User.username == "new_user")).scalar_one()
        assert user.display_name == "A New User"
        assert user.email == "renamed@example.test"
