"""Integration tests for FastAPI Endpoints using Starlette TestClient."""

import pytest
from starlette.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_serve_dashboard():
    """Tests that the main dashboard HTML is served successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert "ANPR Sentry Tactical" in response.text


def test_system_status():
    """Tests GET /api/system/status returns correct system telemetry."""
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "device" in data
    assert "active_engine" in data
    assert "total_detections" in data
    assert "total_watchlist" in data


def test_sample_images_endpoint():
    """Tests GET /api/sample_images returns list of test crops."""
    response = client.get("/api/sample_images")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert isinstance(data["samples"], list)


def test_detections_endpoint():
    """Tests GET /api/detections with limit and filtering."""
    response = client.get("/api/detections?limit=10&filter_type=all")
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "total" in data
    assert isinstance(data["detections"], list)


def test_watchlist_endpoint():
    """Tests GET /api/watchlist returns current watchlists."""
    response = client.get("/api/watchlist")
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)
    assert len(items) >= 2


def test_engine_switch():
    """Tests POST /api/engine/switch toggles recognition engines."""
    # Switch to paddleocr
    res1 = client.post("/api/engine/switch", data={"engine": "paddleocr"})
    assert res1.status_code == 200
    assert res1.json()["active_engine"] == "paddleocr"

    # Switch to qwen2_vl
    res2 = client.post("/api/engine/switch", data={"engine": "qwen2_vl"})
    assert res2.status_code == 200
    assert res2.json()["active_engine"] == "qwen2_vl"

    # Invalid engine
    res_bad = client.post("/api/engine/switch", data={"engine": "invalid_engine"})
    assert res_bad.status_code == 400
