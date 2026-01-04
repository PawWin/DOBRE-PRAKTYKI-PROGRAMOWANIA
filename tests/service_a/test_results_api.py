from fastapi.testclient import TestClient

from services.service_a.app.main import app


def test_create_and_get_result(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    client = TestClient(app)

    payload = {
        "id": "job-123",
        "url": "http://example.com/image.jpg",
        "status": "done",
        "count": 3,
    }

    response = client.post("/results", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["id"] == payload["id"]
    assert data["count"] == payload["count"]
    assert "received_at" in data

    get_response = client.get("/results/job-123")
    assert get_response.status_code == 200
    stored = get_response.json()
    assert stored["id"] == payload["id"]
    assert stored["count"] == payload["count"]


def test_get_missing_result(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    client = TestClient(app)

    response = client.get("/results/missing-id")
    assert response.status_code == 404
