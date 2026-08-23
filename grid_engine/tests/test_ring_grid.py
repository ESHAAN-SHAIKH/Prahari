"""
test_ring_grid.py
=================
Tests for grid_engine/ring_grid.py.
"""

import time
import numpy as np
import pytest

from grid_engine.cell import Cell
from grid_engine.ring_grid import RingGrid, DEFAULT_RING_TIERS
from perception.taxonomy import BASE_CLASSES


def test_tier_assignment_by_range():
    """Verify points at distinct ranges are assigned to corresponding resolution tiers."""
    grid = RingGrid()

    # Points at 5m (Tier 0), 20m (Tier 1), 60m (Tier 2)
    ranges = np.array([5.0, 20.0, 60.0], dtype=np.float64)
    assigned_tiers = grid.assign_tiers(ranges)

    assert assigned_tiers[0] == 0, "5m point should be in Tier 0"
    assert assigned_tiers[1] == 1, "20m point should be in Tier 1"
    assert assigned_tiers[2] == 2, "60m point should be in Tier 2"


def test_boundary_point_goes_to_inner_tier():
    """Verify points exactly at tier boundaries land in the inner (finer) tier."""
    grid = RingGrid()

    # Exact boundary points: 10.0m (boundary of 0/1), 30.0m (boundary of 1/2)
    boundary_ranges = np.array([10.0, 10.0001, 30.0, 30.0001], dtype=np.float64)
    assigned_tiers = grid.assign_tiers(boundary_ranges)

    assert assigned_tiers[0] == 0, "Point at exactly 10.0m must land in Tier 0 (inner)"
    assert assigned_tiers[1] == 1, "Point at 10.0001m must land in Tier 1"
    assert assigned_tiers[2] == 1, "Point at exactly 30.0m must land in Tier 1 (inner)"
    assert assigned_tiers[3] == 2, "Point at 30.0001m must land in Tier 2"


def test_class_priority_vote():
    """Verify hazard priority class voting ensures obstacles override terrain majority."""
    grid = RingGrid()

    # Create points in the same cell: (x=2.01, y=2.01) falling in Tier 0 cell
    # 99 points of terrain (0) + 1 point of pedestrian (2)
    N = 100
    points = np.full((N, 3), [2.01, 2.01, 0.0], dtype=np.float32)
    class_ids = np.zeros(N, dtype=np.int32)  # terrain
    class_ids[42] = 2                       # single pedestrian point

    cells = grid.project(points, class_ids)
    assert len(cells) == 1

    cell = list(cells.values())[0]
    assert cell.point_count == 100
    assert cell.class_id == 2, "Pedestrian hazard class must win over terrain plurality!"


def test_projection_throughput_and_cell_properties():
    """Verify vectorized projection of 100k points is fast and computes correct stats."""
    grid = RingGrid()
    rng = np.random.default_rng(42)

    # 100k synthetic points
    N = 100_000
    pts = rng.uniform(-50.0, 50.0, (N, 3)).astype(np.float32)
    pts[:, 2] = rng.uniform(-1.5, 3.0, N).astype(np.float32)
    classes = rng.choice([0, 1, 2, 3, 255], size=N, p=[0.5, 0.2, 0.1, 0.1, 0.1]).astype(np.int32)

    t0 = time.perf_counter()
    cells = grid.project(pts, classes)
    dt = time.perf_counter() - t0

    print(f"\nVectorized 100k projection time: {dt * 1000.0:.2f} ms (Cells: {len(cells)})")
    
    assert len(cells) > 0
    # Must comfortably meet real-time frame budget on laptop CPU (< 500 ms for 100k points)
    assert dt < 0.500, f"Projection took too long: {dt:.4f}s"

    # Verify cell properties
    sample_cell = list(cells.values())[0]
    assert isinstance(sample_cell, Cell)
    assert round(sample_cell.size, 2) in (0.05, 0.15, 0.50)
    assert sample_cell.height_min <= sample_cell.height_mean <= sample_cell.height_max
    assert sample_cell.height_var >= 0.0
    assert sample_cell.point_count > 0
