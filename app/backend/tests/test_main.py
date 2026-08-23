"""
test_main.py
============
Tests for app/backend/main.py.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "prahari-lite-backend"
    assert "version" in data


def test_status_endpoint():
    """Verify GET /status returns online status, active backend, and ring tiers."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "total_frames" in data
    assert "active_backend" in data
    assert "ring_tiers" in data
    assert len(data["ring_tiers"]) == 3


def test_cors_headers():
    """Verify CORS headers allow cross-origin requests from localhost."""
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in ("http://localhost:3000", "*")
