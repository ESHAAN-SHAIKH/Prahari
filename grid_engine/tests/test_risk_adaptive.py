"""
test_risk_adaptive.py
=====================
⭐ Tests for TASK-010: Risk-Adaptive Local Subdivision Rule.

Validates that high height-variance hazards (bumps, potholes, obstacles)
in distant tiers trigger local subdivision into finer-resolution cells,
upgrading the system from purely distance-tiered to genuinely risk-adaptive.
"""

from pathlib import Path
import numpy as np
import pytest

from grid_engine.ring_grid import RingGrid


def test_far_bump_subdivides_finer_than_close_flat_road():
    """
    Core Novelty Regression Test (TASK-010):
    A flat road patch at 8m (near-field) vs a bump/pothole at 20m (mid-range).

    Under pure distance rules:
      - 8m flat road -> Tier 0 (0.05m / 5cm)
      - 20m bump -> Tier 1 (0.15m / 15cm)

    Under risk-adaptive rule:
      - 20m bump has high height-variance -> subdivided into Tier 0 (0.05m / 5cm) sub-cells!
      - Result: The far bump ends up with 0.05m cells, matching the near-field resolution.
    """
    grid = RingGrid()

    # 1. Close flat road patch at (x=6.0, y=0.0) -> r = 6.0m (Tier 0)
    # 50 points on flat surface: z = 0.0 +- 0.001
    n_flat = 50
    flat_x = np.random.uniform(5.8, 6.2, n_flat)
    flat_y = np.random.uniform(-0.2, 0.2, n_flat)
    flat_z = np.random.normal(0.0, 0.001, n_flat)
    flat_pts = np.column_stack([flat_x, flat_y, flat_z])
    flat_cls = np.zeros(n_flat, dtype=np.int32)

    # 2. Far bump/pothole patch at (x=20.0, y=0.0) -> r = 20.0m (Tier 1)
    # 50 points with sharp step height variation: z in [-0.2, +0.3] -> high variance
    n_bump = 50
    bump_x = np.random.uniform(19.8, 20.2, n_bump)
    bump_y = np.random.uniform(-0.2, 0.2, n_bump)
    bump_z = np.random.choice([-0.20, +0.30], size=n_bump)  # Step obstacle / pothole edge
    bump_pts = np.column_stack([bump_x, bump_y, bump_z])
    bump_cls = np.ones(n_bump, dtype=np.int32)

    # Combined scene
    scene_pts = np.vstack([flat_pts, bump_pts]).astype(np.float32)
    scene_cls = np.concatenate([flat_cls, bump_cls])

    # A) With pure distance mode (default)
    cells_standard = grid.project(scene_pts, scene_cls, enable_risk_adaptive=False)
    bump_cells_std = [c for c in cells_standard.values() if c.x > 15.0]
    assert all(np.isclose(c.size, 0.15) for c in bump_cells_std), (
        "In pure distance mode, 20m bump must be in 0.15m Tier 1 cells"
    )

    # B) With risk-adaptive mode ENABLED
    cells_adaptive = grid.project(
        scene_pts,
        scene_cls,
        enable_risk_adaptive=True,
        variance_threshold=0.02,
    )
    bump_cells_adapt = [c for c in cells_adaptive.values() if c.x > 15.0]

    # The high-variance 20m bump must be subdivided into finer Tier 0 (0.05m) cells!
    assert any(np.isclose(c.size, 0.05) for c in bump_cells_adapt), (
        "Risk-adaptive rule must subdivide high-variance 20m bump into 0.05m sub-cells!"
    )


def test_flat_distant_road_is_not_subdivided():
    """Verify that flat surfaces in distant tiers remain coarse to preserve memory savings."""
    grid = RingGrid()

    # Flat road at 25m (low variance)
    n_pts = 100
    pts_x = np.random.uniform(24.5, 25.5, n_pts)
    pts_y = np.random.uniform(-0.5, 0.5, n_pts)
    pts_z = np.random.normal(0.0, 0.001, n_pts)  # Very flat road
    pts = np.column_stack([pts_x, pts_y, pts_z]).astype(np.float32)
    cls = np.zeros(n_pts, dtype=np.int32)

    cells = grid.project(pts, cls, enable_risk_adaptive=True, variance_threshold=0.04)

    # All cells at 25m must stay at Tier 1 (0.15m) because height variance is minimal
    for c in cells.values():
        assert np.isclose(c.size, 0.15)
        assert c.tier == 1
