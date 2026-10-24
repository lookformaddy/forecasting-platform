"""FastAPI surface — auto-skips cleanly when FastAPI is absent."""

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from forecasting.service import create_app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(create_app())


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_forecast_endpoint(client):
    payload = {
        "series_name": "demand",
        "values": [100.0 + i * 0.5 + (5.0 if i % 7 == 0 else 0.0) for i in range(60)],
        "start_date": "2026-01-01",
        "seasonal_period": 7,
        "model": "holt-winters",
        "horizon": 7,
    }
    resp = client.post("/forecast", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["forecasts"]) == 7
    assert "interval" in body


def test_forecast_endpoint_rejects_unknown_model(client):
    payload = {"series_name": "x", "values": [1.0] * 30, "model": "nope", "horizon": 3}
    assert client.post("/forecast", json=payload).status_code == 422
