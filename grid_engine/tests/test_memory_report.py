"""
test_memory_report.py
=====================
Tests for grid_engine/uniform_grid.py and grid_engine/memory_report.py.
"""

from pathlib import Path
import numpy as np
import pytest

from grid_engine.ring_grid import RingGrid
from grid_engine.uniform_grid import UniformGrid
from grid_engine.memory_report import evaluate_frame_memory, generate_memory_report
from perception.loader import get_frame_count


def test_uniform_grid_cell_count_matches_naive_formula():
    """Verify uniform grid active cell count is bounded by theoretical max (200m/0.05m)^2 = 16M cells."""
    ug = UniformGrid(cell_size_m=0.05)
    pts = np.random.uniform(-50.0, 50.0, (10_000, 3)).astype(np.float32)
    classes = np.zeros(len(pts), dtype=np.int32)

    cells = ug.project(pts, classes, max_range=100.0)
    # Total cells must be <= number of input points and <= 16,000,000
    assert len(cells) <= len(pts)
    assert len(cells) <= 16_000_000

    # Every cell in uniform grid has size 0.05
    for c in cells.values():
        assert np.isclose(c.size, 0.05)


def test_adaptive_always_smaller_or_equal():
    """Invariant: Adaptive ring grid cell count must NEVER exceed Uniform grid on any frame."""
    num_frames = min(5, get_frame_count())
    assert num_frames > 0

    for f_idx in range(num_frames):
        res = evaluate_frame_memory(f_idx)
        assert res["adaptive_cells"] <= res["uniform_cells"], (
            f"Frame {f_idx} failed invariant: Adaptive {res['adaptive_cells']} > Uniform {res['uniform_cells']}"
        )
        assert res["memory_saved_pct"] >= 0.0


def test_evaluate_frame_memory_structure():
    """Verify returned metrics structure and values."""
    res = evaluate_frame_memory(0)
    assert "frame_idx" in res
    assert "point_count" in res
    assert "uniform_cells" in res
    assert "adaptive_cells" in res
    assert "memory_saved_pct" in res
    assert "uniform_mb" in res
    assert "adaptive_mb" in res

    assert res["point_count"] > 0
    assert res["uniform_cells"] > 0
    assert res["adaptive_cells"] > 0
    assert 0.0 <= res["memory_saved_pct"] <= 100.0
