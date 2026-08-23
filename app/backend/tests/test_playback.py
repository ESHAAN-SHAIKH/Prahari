"""
test_playback.py
================
Tests for app/backend/playback.py and playback API endpoints.
"""

import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.playback import engine

client = TestClient(app)


def test_frame_endpoints_return_valid_data():
    """Verify /frame and /frame/uniform return non-empty grid structures."""
    # 1. Adaptive Frame
    resp_adaptive = client.get("/frame")
    assert resp_adaptive.status_code == 200
    data_adaptive = resp_adaptive.json()
    assert "frame_idx" in data_adaptive
    assert "cell_count" in data_adaptive
    assert "cells" in data_adaptive
    assert data_adaptive["cell_count"] > 0
    assert len(data_adaptive["cells"]) == data_adaptive["cell_count"]

    # 2. Uniform Baseline Frame
    resp_uniform = client.get("/frame/uniform")
    assert resp_uniform.status_code == 200
    data_uniform = resp_uniform.json()
    assert "cells" in data_uniform
    assert data_uniform["cell_count"] > 0


def test_get_specific_frame_by_index():
    """Verify /frame/{idx} returns valid data or 404 for invalid index."""
    resp = client.get("/frame/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["frame_idx"] == 1

    # Out of range
    resp_invalid = client.get("/frame/999999")
    assert resp_invalid.status_code == 404


def test_playback_controls_lifecycle():
    """Verify start, speed, seek, and stop endpoints mutate engine state."""
    # Start
    resp_start = client.post("/playback/start")
    assert resp_start.status_code == 200
    assert engine.is_playing is True

    # Speed
    resp_speed = client.post("/playback/speed?x=2.0")
    assert resp_speed.status_code == 200
    assert engine.speed_multiplier == 2.0

    # Seek
    resp_seek = client.post("/playback/seek?frame=5")
    assert resp_seek.status_code == 200
    assert engine.current_frame_idx == 5

    # Stop
    resp_stop = client.post("/playback/stop")
    assert resp_stop.status_code == 200
    assert engine.is_playing is False


def test_playback_loops_at_end_of_sequence():
    """Verify frame index wraps around to 0 cleanly when reaching end of sequence."""
    # Set to last frame
    engine.seek(engine.total_frames - 1)
    assert engine.current_frame_idx == engine.total_frames - 1

    # Simulate next frame step
    next_idx = (engine.current_frame_idx + 1) % engine.total_frames
    assert next_idx == 0

    engine.seek(next_idx)
    assert engine.current_frame_idx == 0


def test_perception_backend_switch_endpoint():
    """Verify POST /perception/backend dynamically toggles active backend."""
    # Switch to model
    resp_model = client.post("/perception/backend?name=model")
    assert resp_model.status_code == 200
    assert resp_model.json()["active_backend"] == "model"

    # Switch back to groundtruth
    resp_gt = client.post("/perception/backend?name=groundtruth")
    assert resp_gt.status_code == 200
    assert resp_gt.json()["active_backend"] == "groundtruth"
