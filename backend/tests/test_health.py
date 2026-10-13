from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_error_format() -> None:
    r = TestClient(app).get("/nope")
    assert r.status_code == 404
    assert "code" in r.json()["error"]
