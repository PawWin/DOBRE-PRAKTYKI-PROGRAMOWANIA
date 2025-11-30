from app.models import User


def test_create_user_with_admin_role(authorized_client, db_session):
    payload = {
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "pass12345",
        "roles": ["ROLE_USER"],
    }

    response = authorized_client.post("/users", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == payload["username"]
    assert "ROLE_USER" in data["roles"]

    stored = db_session.query(User).filter(User.username == "newuser").one()
    assert stored.email == payload["email"]


def test_create_user_without_admin_permissions(client, user_factory):
    user, password = user_factory(roles=["ROLE_USER"], password="basicpass")
    login_response = client.post("/login", json={"username": user.username, "password": password})
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})

    payload = {
        "username": "forbidden",
        "email": "forbidden@example.com",
        "password": "pass12345",
        "roles": ["ROLE_USER"],
    }

    response = client.post("/users", json=payload)
    assert response.status_code == 403
    assert response.json()["detail"] == "Insufficient permissions"
