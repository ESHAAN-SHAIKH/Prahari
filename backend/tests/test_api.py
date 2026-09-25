import json

import pytest
from fastapi.testclient import TestClient

from backend import codec
from backend import config as C
from backend.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_system_describes_the_geometry_the_ps_asks_for():
    s = client.get("/api/system").json()
    g = s["geometry"]
    assert g["min_cell_m"] == pytest.approx(0.05)
    assert g["fovea_radius_m"] == 10.0
    assert g["max_range_m"] == 100.0
    assert g["coarse_size_m"] == 0.50
    # The level table must actually span the schedule, not just claim it.
    sizes = [row["size_m"] for row in s["level_table"]]
    assert min(sizes) == pytest.approx(0.05)
    assert any(abs(sz - 0.4) < 1e-9 for sz in sizes)
    assert set(s["ps_categories"]) == {"walls", "poles", "pedestrians", "vehicles"}


def test_system_reports_its_perception_source_honestly():
    p = client.get("/api/system").json()["perception"]
    assert p["source"] in ("synthetic", "checkpoint")
    if p["source"] == "synthetic":
        assert "generated" in (p["notes"] or "").lower() or "analytic" in (p["notes"] or "").lower()


def test_frame_endpoint_returns_a_decodable_frame():
    r = client.get("/api/frame")
    assert r.status_code == 200
    d = codec.decode(r.content)
    assert d["n_cells"] > 0
    meta = json.loads(r.headers["X-Prahari-Meta"])
    assert meta["wire_bytes"] == len(r.content)
    tail = d["tail"]
    assert tail["source"] in ("synthetic", "checkpoint")
    assert tail["stats"]["n_cells"] == d["n_cells"]
    assert tail["timing"]["frame_ms"] > 0


def test_frame_endpoint_honours_risk_adaptive_flag():
    on = codec.decode(client.get("/api/frame?risk_adaptive=true").content)
    off = codec.decode(client.get("/api/frame?risk_adaptive=false").content)
    assert off["n_cells"] < on["n_cells"]
    assert set(off["driver"].tolist()) == {C.DRIVER_RANGE}


def test_compare_endpoint_quantifies_the_novelty():
    c = client.get("/api/compare").json()
    assert c["risk_adaptive"]["median_hazard_cell_m"] < c["range_only"]["median_hazard_cell_m"]
    assert c["risk_adaptive"]["hazard_cells_beyond_fovea"] > 0
    assert c["dense_uniform_5cm"]["n_cells"] == 16_000_000
    assert c["risk_adaptive"]["memory"]["reduction_vs_dense"] > 100


def test_metrics_endpoint_never_invents_numbers():
    m = client.get("/api/metrics").json()
    for key in ("per_class", "distance", "edge", "baselines"):
        block = m[key]
        assert block["status"] in ("measured", "not_measured")
        if block["status"] == "not_measured":
            assert block["how"]                       # tells you the command to run
            assert "value" not in block               # and offers no stand-in figure


def test_index_is_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "PRAHARI" in r.text


def test_websocket_streams_decodable_frames():
    with client.websocket_connect("/ws/stream") as ws:
        first = codec.decode(ws.receive_bytes())
        assert first["n_cells"] > 0
        second = codec.decode(ws.receive_bytes())
        assert second["frame"] > first["frame"]
        assert second["tail"]["rolling"]["samples"] >= 1


def test_websocket_applies_settings():
    with client.websocket_connect("/ws/stream") as ws:
        codec.decode(ws.receive_bytes())
        ws.send_text(json.dumps({"cmd": "set", "risk_adaptive": False}))
        for _ in range(6):
            d = codec.decode(ws.receive_bytes())
            if d["tail"]["settings"]["risk_adaptive"] is False:
                assert set(d["driver"].tolist()) == {C.DRIVER_RANGE}
                return
        pytest.fail("risk_adaptive=False was never applied")


def test_stream_keeps_up_with_ten_hertz():
    """The PS requires the classify + grid-update loop at >=10 Hz."""
    with client.websocket_connect("/ws/stream") as ws:
        last = None
        for _ in range(12):
            last = codec.decode(ws.receive_bytes())
        roll = last["tail"]["rolling"]
        assert roll["frame_ms_mean"] < 100.0, f"frame took {roll['frame_ms_mean']} ms"
        assert roll["fps"] >= 10.0
