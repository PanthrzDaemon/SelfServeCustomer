from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)


def test_root_identifies_service():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == settings.app_name
    assert body["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_uses_configured_title():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["info"]["title"] == settings.app_name
