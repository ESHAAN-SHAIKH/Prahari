import json

import numpy as np
import pytest

from backend import codec
from backend import config as C
from backend.grid_engine import GridEngine
from backend.metrics import Evidence
from backend.perception import SyntheticPerception, build_perception


@pytest.fixture(scope="module")
def grid():
    f = SyntheticPerception().infer(5)
    return GridEngine().build(f["xyz"], f["cls"], f["conf"])


# ------------------------------------------------------------------- codec ----
def test_roundtrip_is_lossless_where_it_must_be(grid):
    wire, _ = codec.encode(grid, 42, {"stats": grid.stats})
    d = codec.decode(wire)
    assert d["frame"] == 42 and d["n_cells"] == len(grid)
    # Geometry and labels must survive — an off-by-one cell is a map error. x/y are
    # carried as exact integer multiples of the quantisation step, so the only residue
    # is float32 representation, far below a millimetre.
    assert np.abs(d["x"] - grid.x).max() < 1e-3
    assert np.abs(d["y"] - grid.y).max() < 1e-3
    assert np.array_equal(np.rint(d["x"] / C.QUANT_M), np.rint(grid.x / C.QUANT_M))
    assert np.array_equal(d["level"], grid.level)
    assert np.array_equal(d["cls"], grid.cls)
    assert np.array_equal(d["driver"], grid.driver)


def test_quantisation_error_is_bounded(grid):
    d = codec.decode(codec.encode(grid, 0, {})[0])
    assert np.abs(d["z"] - grid.z).max() <= 0.005 + 1e-6     # 1 cm buckets, half-cm error
    assert np.abs(d["conf"] - grid.conf).max() <= 1.0 / 255 + 1e-6


def test_tail_survives(grid):
    tail = {"stats": grid.stats, "detections": [{"cls": 7, "r": 9.5}], "source": "synthetic"}
    d = codec.decode(codec.encode(grid, 1, tail)[0])
    assert d["tail"]["source"] == "synthetic"
    assert d["tail"]["detections"][0]["cls"] == 7


def test_compression_shrinks_the_frame(grid):
    _, comp = codec.encode(grid, 0, {"stats": grid.stats}, compress=True)
    _, raw = codec.encode(grid, 0, {"stats": grid.stats}, compress=False)
    assert comp["wire_bytes"] < raw["wire_bytes"]
    assert comp["compression_ratio"] > 1.5


def test_uncompressed_also_decodes(grid):
    wire, _ = codec.encode(grid, 7, {}, compress=False)
    assert codec.decode(wire)["n_cells"] == len(grid)


def test_bad_magic_is_rejected():
    with pytest.raises(ValueError):
        codec.decode(b"XXXX" + bytes(64))


def test_link_budget_is_arithmetic_not_optimism(grid):
    _, info = codec.encode(grid, 0, {"stats": grid.stats})
    b = codec.link_budget(info["wire_bytes"], 10.0)
    assert b["kbps"] == pytest.approx(info["wire_bytes"] * 8 * 10 / 1000.0, abs=0.05)
    assert b["fits_1mbps_link"] == (info["wire_bytes"] * 80 <= 1_000_000)


def test_wire_size_beats_the_raw_cloud(grid):
    _, info = codec.encode(grid, 0, {"stats": grid.stats})
    assert info["wire_bytes"] < grid.stats["memory"]["raw_cloud_bytes"]


# -------------------------------------------------------------- perception ----
def test_synthetic_frame_is_well_formed():
    f = SyntheticPerception().infer(11)
    n = f["xyz"].shape[0]
    assert n > 10_000
    assert f["cls"].shape == (n,) and f["conf"].shape == (n,)
    assert f["conf"].min() >= 0.0 and f["conf"].max() <= 1.0
    assert np.isfinite(f["xyz"]).all()
    assert f["source"] == "synthetic"


def test_scene_contains_every_ps_category():
    """Walls, poles, pedestrians and vehicles must all be present, or the per-class
    demonstration has nothing to show."""
    p = SyntheticPerception()
    seen = set()
    for i in range(0, 40, 5):
        seen |= set(np.unique(p.infer(i)["cls"]).tolist())
    for name, ids in C.PS_CATEGORIES.items():
        assert seen & set(ids), f"no points for PS category {name}"
    assert seen & C.HAZARD_IDS, "no hazard points anywhere in the loop"


def test_dust_sector_actually_degrades_confidence():
    f = SyntheticPerception().infer(0)
    az = np.degrees(np.arctan2(f["xyz"][:, 1], f["xyz"][:, 0]))
    d = np.abs(((az - f["dust_center_deg"] + 180) % 360) - 180)
    inside, outside = f["conf"][d < 15], f["conf"][d > 90]
    assert inside.mean() < outside.mean() - 0.1


def test_objects_move_between_frames():
    p = SyntheticPerception()
    a = {(d["cls"], d["r"]) for d in p.infer(0)["detections"]}
    b = {(d["cls"], d["r"]) for d in p.infer(30)["detections"]}
    assert a != b


def test_partial_checkpoint_config_does_not_silently_pass(monkeypatch):
    """Half-configured checkpoint mode must fall back *and say so* — a silent fallback is
    how a demo ends up showing synthetic data while everyone believes it is live."""
    monkeypatch.setattr(C, "CHECKPOINT_PATH", "/tmp/nope.pth")
    monkeypatch.setattr(C, "CHECKPOINT_CONFIG", "")
    monkeypatch.setenv("PRAHARI_SCANS", "")
    perception, note = build_perception()
    assert perception.source == "synthetic"
    assert note and "PRAHARI_CFG" in note


# ---------------------------------------------------------------- evidence ----
def test_missing_evidence_reports_not_measured(tmp_path):
    ev = Evidence(tmp_path / "absent.json")
    assert not ev.available
    for block in (ev.per_class(), ev.distance(), ev.edge(), ev.baselines()):
        assert block["status"] == "not_measured"
        assert block["how"], "an unmeasured block must say how to produce it"


def test_measured_evidence_is_read_with_provenance(tmp_path):
    export = {
        "generated": "2026-01-01T00:00:00+00:00",
        "edge_board": "Jetson Orin Nano 8GB",
        "data_caveats": ["mini-set validation"],
        "checks": {"P4.2": {"text": "distance chart", "done": True},
                   "P6.3": {"text": "latency measured", "done": False}},
        "ledger": [
            {"metric": "frnet_indian_latency_ms", "value": 62.4, "unit": "ms", "tag": "edge",
             "command": "trtexec --loadEngine=...", "hardware": "Orin Nano; JetPack 6.0"},
            {"metric": "frnet_indian_fps", "value": 16.0, "unit": "FPS", "tag": "edge",
             "command": "trtexec --loadEngine=...", "hardware": "Orin Nano; JetPack 6.0"},
        ],
        "baselines": {"stock_backbone": {"name": "Stock FRNet", "checkpoint": "/ckpt/a.pth"}},
    }
    p = tmp_path / "trinetra_evidence_export.json"
    p.write_text(json.dumps(export))
    (tmp_path / "distance_stratified_frnet.json").write_text(json.dumps({
        "label": "frnet_indian_finetune", "n_scans": 120, "command": "run_distance_eval(...)",
        "summary": {"0-10m": {"mIoU_dataset": 71.2}, "30-100m": {"mIoU_dataset": 48.9}}}))

    ev = Evidence(p)
    assert ev.available
    edge = ev.edge()
    assert edge["status"] == "measured"
    model = edge["models"][0]
    assert model["fps"] == 16.0 and model["command"].startswith("trtexec")
    dist = ev.distance()
    assert dist["status"] == "measured"
    assert dist["runs"][0]["summary"]["0-10m"]["mIoU_dataset"] == 71.2
    assert ev.baselines()["status"] == "measured"
    cl = ev.checklist()
    assert cl["done"] == 1 and cl["total"] == 2


def test_selftest_evidence_is_ignored(tmp_path):
    """Self-test output from the notebook must never be mistaken for a result."""
    p = tmp_path / "trinetra_evidence_export.json"
    p.write_text(json.dumps({"ledger": []}))
    (tmp_path / "distance_stratified_SELFTEST_synthetic.json").write_text(
        json.dumps({"label": "SELFTEST_synthetic", "summary": {"0-10m": {"mIoU_dataset": 99}}}))
    assert Evidence(p).distance()["status"] == "not_measured"
