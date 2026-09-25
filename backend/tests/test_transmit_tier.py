import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import codec
from backend import config as C
from backend.grid_engine import GridEngine, decimate
from backend.main import app
from backend.perception import SyntheticPerception

client = TestClient(app)


@pytest.fixture(scope="module")
def full():
    f = SyntheticPerception().infer(3)
    return GridEngine().build(f["xyz"], f["cls"], f["conf"])


def test_tier_shrinks_the_frame(full):
    for level in (11, 10, 9, 8):
        d = decimate(full, level)
        assert len(d) < len(full)
        assert d.size.max() <= C.ROOT_SIZE_M / 2 ** level + 1e-9 or \
            (d.driver != C.DRIVER_RANGE).any()


def test_tier_loses_no_points(full):
    """Coarsening merges cells; it must not drop measurements."""
    for level in (11, 9):
        d = decimate(full, level)
        assert int(d.n_points.sum()) == int(full.n_points.sum())


def test_tier_preserves_risk_cells(full):
    """The whole point: the remote node gets a coarse road and a full-resolution pothole."""
    d = decimate(full, 9, preserve_risk=True)
    plain = decimate(full, 9, preserve_risk=False)

    def haz(g):
        m = np.isin(g.cls, list(C.HAZARD_IDS)) & (np.hypot(g.x, g.y) > C.FOVEA_RADIUS_M)
        return g.size[m]

    kept, dropped = haz(d), haz(plain)
    assert kept.size and dropped.size
    assert np.median(kept) < np.median(dropped)
    assert d.stats["decimation"]["preserved_risk_cells"] > 0


def test_tier_cuts_bandwidth_enough_to_matter(full):
    _, before = codec.encode(full, 0, {})
    _, after = codec.encode(decimate(full, 9), 0, {})
    assert after["wire_bytes"] < before["wire_bytes"] / 3
    budget = codec.link_budget(after["wire_bytes"], 10.0)
    assert budget["mbps"] < codec.link_budget(before["wire_bytes"], 10.0)["mbps"]


def test_tier_elevation_stays_consistent(full):
    d = decimate(full, 9)
    assert np.all(d.z_min <= d.z + 1e-4)
    assert np.all(d.z <= d.z_max + 1e-4)
    assert d.z_min.min() == pytest.approx(full.z_min.min(), abs=1e-4)


def test_tier_is_a_no_op_above_the_finest_level(full):
    d = decimate(full, C.MAX_DEPTH)
    assert len(d) == len(full)
    assert d.stats["decimation"]["merged_cells"] == 0


def test_tier_reports_what_it_dropped(full):
    d = decimate(full, 9)
    info = d.stats["decimation"]
    assert info["cells_before"] == len(full)
    assert info["cells_after"] == len(d)
    assert info["merged_from"] > info["merged_into"]
    assert info["max_cell_m"] == pytest.approx(0.4)


def test_api_frame_applies_the_tier():
    a = codec.decode(client.get("/api/frame").content)
    b = codec.decode(client.get("/api/frame?link_tier=9").content)
    assert b["n_cells"] < a["n_cells"]
    assert b["tail"]["settings"]["link_tier"] == 9
    # The full onboard map is still reported alongside what was transmitted.
    assert b["tail"]["onboard"]["n_cells"] > b["n_cells"]


def test_compare_endpoint_lists_the_tiers():
    tiers = client.get("/api/compare").json()["transmit_tiers"]
    assert "level9_risk_preserved" in tiers and "level9_plain" in tiers
    preserved = tiers["level9_risk_preserved"]
    plain = tiers["level9_plain"]
    assert preserved["median_hazard_cell_m"] < plain["median_hazard_cell_m"]
    assert preserved["wire_bytes"] >= plain["wire_bytes"]   # keeping risk costs bytes
    assert preserved["mbps_at_10hz"] < 4.0
