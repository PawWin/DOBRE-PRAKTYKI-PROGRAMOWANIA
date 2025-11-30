import pytest


def test_login_success(client, admin_credentials):
    response = client.post("/login", json=admin_credentials)
    assert response.status_code == 200
    body = response.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


def test_login_invalid_credentials(client, admin_credentials):
    payload = {**admin_credentials, "password": "wrong-pass"}
    response = client.post("/login", json=payload)
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid username or password"


def test_user_details_with_valid_token(authorized_client, admin_credentials):
    response = authorized_client.get("/user_details")
    assert response.status_code == 200
    body = response.json()
    assert body["sub"] == admin_credentials["username"]
    assert "roles" in body


def test_user_details_missing_token(client):
    response = client.get("/user_details")
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing Bearer token"
