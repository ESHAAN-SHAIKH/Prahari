"""
test_conservation.py
====================
🔬 Alignment & Point-Conservation Test Suite for RingGrid.

Directly proves the core problem statement requirement:
"without causing alignment errors or data loss".
"""

from pathlib import Path
import numpy as np
import pytest

from grid_engine.ring_grid import RingGrid
from perception.loader import get_classified_points, get_frame_count


@pytest.fixture
def grid():
    return RingGrid()


class TestPointConservation:
    """1. Total points in cells must equal valid input points (0% data loss)."""

    def test_synthetic_point_count_conserved(self, grid):
        rng = np.random.default_rng(101)
        N = 50_000
        # Generate points strictly within 100m max_range
        pts = rng.uniform(-60.0, 60.0, (N, 3)).astype(np.float32)
        r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        valid_mask = r <= 100.0
        pts = pts[valid_mask]
        classes = rng.choice([0, 1, 2, 3, 255], size=len(pts)).astype(np.int32)

        cells = grid.project(pts, classes, max_range=100.0)
        total_cell_points = sum(cell.point_count for cell in cells.values())

        assert total_cell_points == len(pts), (
            f"Point conservation failed: {total_cell_points} in cells vs {len(pts)} inputs"
        )

    def test_demo_sequence_point_count_conserved(self, grid):
        num_frames = min(10, get_frame_count())
        assert num_frames > 0, "No frames in demo sequence to test"

        for f_idx in range(num_frames):
            pts, classes = get_classified_points(f_idx)
            r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
            expected_count = int(np.sum(r <= 100.0))

            cells = grid.project(pts, classes, max_range=100.0)
            total_cell_points = sum(cell.point_count for cell in cells.values())

            assert total_cell_points == expected_count, (
                f"Frame {f_idx} point conservation failed: {total_cell_points} vs {expected_count}"
            )


class TestSpatialAlignmentBounds:
    """2. Every point must reside strictly within its assigned cell's bounding box."""

    def test_point_within_assigned_cell_bounds(self, grid):
        rng = np.random.default_rng(202)
        N = 20_000
        pts = rng.uniform(-50.0, 50.0, (N, 3)).astype(np.float32)
        r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        pts = pts[r <= 100.0]
        classes = np.zeros(len(pts), dtype=np.int32)

        cells = grid.project(pts, classes, max_range=100.0)

        # Sample 500 points and check their containing cell
        sample_indices = rng.choice(len(pts), size=min(500, len(pts)), replace=False)
        for idx in sample_indices:
            px, py = pts[idx, 0], pts[idx, 1]
            pr = np.sqrt(px ** 2 + py ** 2)
            tier = grid.assign_tiers(np.array([pr]))[0]
            c_size = grid.tiers[tier]["cell_size_m"]
            ix = int(np.floor(px / c_size))
            iy = int(np.floor(py / c_size))

            key = (tier, ix, iy)
            assert key in cells, f"Point ({px}, {py}) key {key} missing from projected cells"

            cell = cells[key]
            half_size = cell.size / 2.0 + 1e-6  # small epsilon for float precision

            assert (cell.x - half_size) <= px <= (cell.x + half_size), (
                f"Point x={px} outside cell x-bounds [{cell.x - half_size}, {cell.x + half_size}]"
            )
            assert (cell.y - half_size) <= py <= (cell.y + half_size), (
                f"Point y={py} outside cell y-bounds [{cell.y - half_size}, {cell.y + half_size}]"
            )


class TestProperDisjointPartition:
    """3. No point is assigned to multiple cells (proper disjoint partition)."""

    def test_no_double_assignment(self, grid):
        pts, classes = get_classified_points(0)
        r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        valid_pts = pts[r <= 100.0]
        valid_cls = classes[r <= 100.0]

        cells = grid.project(valid_pts, valid_cls, max_range=100.0)

        # Sum of counts across all cells must equal exactly number of valid points
        total_counts = sum(cell.point_count for cell in cells.values())
        assert total_counts == len(valid_pts)

        # Keys in cell dict must be unique
        assert len(cells.keys()) == len(set(cells.keys()))
