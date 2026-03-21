from fastapi.testclient import TestClient

from src.api.main import app


def test_health_endpoint_reachable():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
