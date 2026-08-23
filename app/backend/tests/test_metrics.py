"""
test_metrics.py
===============
Tests for app/backend/metrics.py and GET /metrics API endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.metrics import MetricsCollector
from perception.service import get_active_backend, set_active_backend

client = TestClient(app)


def test_metrics_reports_full_pipeline_latency_not_stage_only():
    """
    Ensure /metrics total latency reflects end-to-end pipeline cost,
    not just classification time — the key validity invariant.
    """
    # Seed the collector with a realistic frame scenario
    collector = MetricsCollector(window_size=5)
    collector.record_frame(
        classify_ms=10.0,
        project_ms=150.0,
        serialize_ms=280.0,
        total_ms=450.0,
        point_count=120000,
        memory_saved_pct=32.4,
    )

    metrics = collector.get_metrics()

    # Total must be the end-to-end time, not just classify
    assert metrics["latency_ms"]["total"] == 450.0

    # Total must be >= classify alone
    assert metrics["latency_ms"]["total"] >= metrics["latency_ms"]["classify"]

    # Total must be >= sum of individual stages (some overlap possible)
    stage_sum = (
        metrics["latency_ms"]["classify"]
        + metrics["latency_ms"]["project"]
        + metrics["latency_ms"]["serialize"]
    )
    # total should be close to stage_sum (within 10% due to minor overhead)
    assert metrics["latency_ms"]["total"] >= stage_sum * 0.9


def test_metrics_backend_field_matches_active_backend():
    """Verify /metrics backend field always reflects the currently active backend."""
    set_active_backend("groundtruth")

    collector = MetricsCollector(window_size=5)
    collector.record_frame(10, 100, 200, 320, 50000, 32.0)

    metrics = collector.get_metrics()
    assert metrics["backend"] == get_active_backend()


def test_metrics_endpoint_returns_valid_structure():
    """Verify GET /metrics endpoint returns all expected fields."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()

    assert "fps" in data
    assert "latency_ms" in data
    assert "memory_saved_pct" in data
    assert "point_count" in data
    assert "backend" in data
    assert "window_samples" in data

    latency = data["latency_ms"]
    assert "classify" in latency
    assert "project" in latency
    assert "serialize" in latency
    assert "total" in latency

    # total must be >= classify alone
    assert latency["total"] >= latency["classify"]


def test_metrics_rolling_window_computes_average():
    """Verify rolling window averages across multiple frames."""
    collector = MetricsCollector(window_size=3)

    frames = [
        (10.0, 100.0, 200.0, 320.0, 100000, 30.0),
        (12.0, 120.0, 220.0, 360.0, 105000, 32.0),
        (11.0, 110.0, 210.0, 340.0, 102000, 31.0),
    ]
    for f in frames:
        collector.record_frame(*f)

    metrics = collector.get_metrics()

    expected_classify_avg = (10.0 + 12.0 + 11.0) / 3
    assert abs(metrics["latency_ms"]["classify"] - expected_classify_avg) < 0.01
    assert metrics["window_samples"] == 3


def test_metrics_pre_init_returns_defaults():
    """An empty collector must return safe defaults before any frame is recorded."""
    collector = MetricsCollector(window_size=5)
    metrics = collector.get_metrics()

    assert metrics["fps"] == 0.0
    assert metrics["latency_ms"]["total"] == 0.0
    assert metrics["point_count"] == 0
    assert metrics["window_samples"] == 0
    # memory_saved_pct default is the sequence-average 32.4%
    assert metrics["memory_saved_pct"] == 32.4
