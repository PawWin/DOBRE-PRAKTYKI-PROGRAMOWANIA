from fastapi.testclient import TestClient

from services.service_b.app.endpoints import analyze as analyze_module
from services.service_b.app.main import app


def test_analyze_img_post(monkeypatch):
    captured = []

    def fake_publish(payload):
        captured.append(payload)

    monkeypatch.setattr(analyze_module, "publish_image_task", fake_publish)
    monkeypatch.setenv("SERVICE_A_PUBLIC_URL", "http://results.local")
    client = TestClient(app)

    response = client.post("/analyze_img", json={"url": "http://example.com/image.jpg"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert data["check_url"].startswith("http://results.local/results/")
    assert captured[0]["url"] == "http://example.com/image.jpg"


def test_analyze_img_get(monkeypatch):
    captured = []

    def fake_publish(payload):
        captured.append(payload)

    monkeypatch.setattr(analyze_module, "publish_image_task", fake_publish)
    monkeypatch.setenv("SERVICE_A_PUBLIC_URL", "http://results.local")
    client = TestClient(app)

    response = client.get("/analyze_img", params={"url": "http://example.com/image.jpg"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert data["check_url"].startswith("http://results.local/results/")
    assert captured[0]["url"] == "http://example.com/image.jpg"
