"""
test_serialize.py
=================
Tests for grid_engine/serialize.py.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from grid_engine.ring_grid import RingGrid
from grid_engine.serialize import grid_to_dict, grid_to_json, benchmark_serialization
from perception.loader import get_classified_points


@pytest.fixture
def sample_grid():
    pts, classes = get_classified_points(0)
    rg = RingGrid()
    return rg.project(pts, classes)


def test_serialize_roundtrip_cell_count(sample_grid):
    """Verify JSON output has exactly as many entries as the input grid has cells."""
    json_str = grid_to_json(sample_grid, frame_idx=0)
    data = json.loads(json_str)

    assert data["frame_idx"] == 0
    assert data["cell_count"] == len(sample_grid)
    assert len(data["cells"]) == len(sample_grid)


def test_serialize_payload_fields(sample_grid):
    """Verify payload fields match the frontend rendering contract."""
    data = grid_to_dict(sample_grid, frame_idx=42)

    assert data["frame_idx"] == 42
    assert "cells" in data
    assert len(data["cells"]) > 0

    first_cell = data["cells"][0]
    assert "x" in first_cell
    assert "y" in first_cell
    assert "size" in first_cell
    assert "height" in first_cell
    assert "class_id" in first_cell

    assert isinstance(first_cell["x"], (int, float))
    assert isinstance(first_cell["y"], (int, float))
    assert isinstance(first_cell["size"], (int, float))
    assert isinstance(first_cell["height"], (int, float))
    assert isinstance(first_cell["class_id"], int)


def test_serialize_benchmark_is_fast(sample_grid):
    """Verify serialization time is benchmarked and recorded."""
    bench = benchmark_serialization(sample_grid, num_trials=3)

    assert bench["cell_count"] == len(sample_grid)
    assert bench["avg_serialize_ms"] > 0
    assert bench["payload_kb"] > 0
    assert bench["avg_serialize_ms"] < 400.0, (
        f"Serialization too slow: {bench['avg_serialize_ms']} ms"
    )
