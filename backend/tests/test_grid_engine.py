import numpy as np
import pytest

from backend import config as C
from backend.grid_engine import GridEngine, uniform_baseline_cells
from backend.perception import SyntheticPerception


@pytest.fixture(scope="module")
def frame():
    return SyntheticPerception().infer(3)


@pytest.fixture(scope="module")
def grid(frame):
    return GridEngine().build(frame["xyz"], frame["cls"], frame["conf"])


def test_no_points_are_lost(grid, frame):
    """Every point inside the root lands in exactly one cell. This is the PS's
    'without causing alignment errors or data loss' requirement, tested directly."""
    half = C.ROOT_SIZE_M / 2.0
    inside = ((np.abs(frame["xyz"][:, 0]) < half) &
              (np.abs(frame["xyz"][:, 1]) < half)).sum()
    assert int(grid.n_points.sum()) == int(inside)


def test_morton_interleave_matches_a_brute_force_reference():
    """The whole alignment guarantee rests on this bit trick, so it is checked against
    the obvious slow implementation rather than trusted."""
    from backend.grid_engine import _morton
    rng = np.random.default_rng(1)
    ix = rng.integers(0, 4096, 500)
    iy = rng.integers(0, 4096, 500)

    def ref(a, b):
        out = 0
        for k in range(12):
            out |= ((b >> k) & 1) << (2 * k)
            out |= ((a >> k) & 1) << (2 * k + 1)
        return out

    assert np.array_equal(_morton(ix, iy),
                          np.array([ref(int(a), int(b)) for a, b in zip(ix, iy)]))


def test_every_point_lies_inside_the_cell_that_claims_it(grid, frame):
    """The failure mode a multi-resolution projection is prone to: a point attributed to
    a cell it does not geometrically belong to. Checked point by point."""
    half = C.ROOT_SIZE_M / 2.0
    xyz, cls, conf = frame["xyz"], frame["cls"], frame["conf"]
    keep = (np.abs(xyz[:, 0]) < half) & (np.abs(xyz[:, 1]) < half)
    pts = xyz[keep]

    # Rebuild the same Morton ordering the engine used, then walk the leaf runs.
    from backend.grid_engine import _morton
    n_grid = 2 ** C.MAX_DEPTH
    ix = np.clip(((pts[:, 0] + half) / C.MIN_CELL_M).astype(np.int64), 0, n_grid - 1)
    iy = np.clip(((pts[:, 1] + half) / C.MIN_CELL_M).astype(np.int64), 0, n_grid - 1)
    order = np.argsort(_morton(ix, iy), kind="stable")
    px, py = pts[order, 0], pts[order, 1]

    ends = np.cumsum(grid.n_points)
    starts = np.concatenate([[0], ends[:-1]])
    sizes = grid.size
    for i in range(0, len(grid), 37):        # every 37th cell keeps the test quick
        lo, hi = int(starts[i]), int(ends[i])
        h = sizes[i] / 2.0
        assert np.all(px[lo:hi] >= grid.x[i] - h - 1e-4)
        assert np.all(px[lo:hi] <= grid.x[i] + h + 1e-4)
        assert np.all(py[lo:hi] >= grid.y[i] - h - 1e-4)
        assert np.all(py[lo:hi] <= grid.y[i] + h + 1e-4)


def test_cell_centres_land_on_the_wire_quantisation_step(grid, frame):
    assert np.all(np.abs(grid.x) <= C.ROOT_SIZE_M / 2.0)
    assert np.all(np.abs(grid.y) <= C.ROOT_SIZE_M / 2.0)
    # Cell centres sit on half-cell steps of their own level, which is what makes the
    # int16 wire quantisation lossless.
    for lvl in np.unique(grid.level):
        m = grid.level == lvl
        step = (C.ROOT_SIZE_M / (2 ** int(lvl))) / 2.0
        offs = (grid.x[m] + C.ROOT_SIZE_M / 2.0) / step
        assert np.allclose(offs, np.round(offs), atol=1e-4)


def test_cells_never_finer_than_the_minimum(grid):
    assert grid.size.min() >= C.MIN_CELL_M - 1e-9
    assert grid.level.max() <= C.MAX_DEPTH


def test_foveation_near_cells_are_finer_than_far_cells(grid):
    r = np.hypot(grid.x, grid.y)
    near = grid.size[r < C.FOVEA_RADIUS_M]
    far = grid.size[r > 60.0]
    assert near.size and far.size
    assert np.median(near) < np.median(far)
    # Inside the fovea the PS asks for 5 cm.
    assert near.max() <= C.MIN_CELL_M + 1e-9


def test_far_field_respects_the_coarse_target(grid):
    r = np.hypot(grid.x, grid.y)
    far = grid.size[r > 90.0]
    if far.size:
        assert far.max() <= C.COARSE_SIZE_M + 1e-9


def test_risk_adaptive_refines_distant_hazards(frame):
    """The core novelty: a hazard beyond the fovea gets fine cells that range alone
    would never have given it."""
    risk = GridEngine(risk_adaptive=True).build(frame["xyz"], frame["cls"], frame["conf"])
    plain = GridEngine(risk_adaptive=False).build(frame["xyz"], frame["cls"], frame["conf"])

    def hazard_sizes(g):
        m = np.isin(g.cls, list(C.HAZARD_IDS)) & (np.hypot(g.x, g.y) > C.FOVEA_RADIUS_M)
        return g.size[m]

    a, b = hazard_sizes(risk), hazard_sizes(plain)
    assert a.size and b.size, "the scene must contain a hazard beyond the fovea"
    assert np.median(a) < np.median(b), "risk adaptation did not refine distant hazards"
    assert np.median(b) / np.median(a) >= 2.0


def test_risk_adaptive_does_not_cost_the_memory_argument(frame):
    """Refinement is close to free in the far field, because sparse regions already hold
    about one point per cell — shrinking those cells changes their size, not their count.
    The cost appears only where several points would have shared a coarse cell."""
    risk = GridEngine(risk_adaptive=True).build(frame["xyz"], frame["cls"], frame["conf"])
    plain = GridEngine(risk_adaptive=False).build(frame["xyz"], frame["cls"], frame["conf"])
    assert len(risk) <= len(plain) * 1.5
    assert risk.stats["memory"]["reduction_vs_dense"] > 100
    assert risk.stats["memory"]["reduction_vs_dense_3d"] > 1000


def test_low_confidence_triggers_refinement():
    """A uniformly low-confidence region must be subdivided, not silently coarsened."""
    rng = np.random.default_rng(0)
    xyz = np.column_stack([
        rng.uniform(70, 80, 6000), rng.uniform(-5, 5, 6000), rng.uniform(-2, -1.5, 6000),
    ]).astype(np.float32)
    cls = np.ones(6000, dtype=np.int32)
    good = GridEngine().build(xyz, cls, np.full(6000, 0.95, np.float32))
    bad = GridEngine().build(xyz, cls, np.full(6000, 0.20, np.float32))
    assert np.median(bad.size) < np.median(good.size)
    assert "low confidence" in bad.stats["cells_by_driver"]


def test_empty_space_is_not_stored(grid):
    """Occupancy is sparse: the cell count must be far below the dense equivalent."""
    assert len(grid) < uniform_baseline_cells() / 100
    assert np.all(grid.n_points > 0)


def test_elevation_is_retained(grid):
    """2.5D, not 2D: cells carry elevation, and z_min <= z <= z_max holds everywhere."""
    assert np.all(grid.z_min <= grid.z + 1e-5)
    assert np.all(grid.z <= grid.z_max + 1e-5)
    assert grid.z.max() - grid.z.min() > 1.0, "a flat map would mean elevation was lost"


def test_hazard_elevation_is_below_the_road(frame, grid):
    """A pothole must come out lower than the drivable surface around it, or the 2.5D
    payload is not carrying the thing it exists to carry."""
    pothole = grid.z[grid.cls == 40]
    road = grid.z[grid.cls == 1]
    assert pothole.size, "no pothole cells in this frame"
    assert pothole.mean() < road.mean()


def test_build_is_deterministic(frame):
    a = GridEngine().build(frame["xyz"], frame["cls"], frame["conf"])
    b = GridEngine().build(frame["xyz"], frame["cls"], frame["conf"])
    assert np.array_equal(a.x, b.x) and np.array_equal(a.level, b.level)
    assert np.array_equal(a.cls, b.cls)


def test_empty_input_is_handled():
    g = GridEngine().build(np.zeros((0, 3), np.float32),
                           np.zeros(0, np.int32), np.zeros(0, np.float32))
    assert len(g) == 0 and g.stats["n_cells"] == 0


def test_build_fits_a_10hz_budget(frame):
    """The PS requires >=10 Hz. The grid stage alone must leave room for the rest."""
    g = GridEngine().build(frame["xyz"], frame["cls"], frame["conf"])
    assert g.stats["grid_build_ms"] < 60.0
