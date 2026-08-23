"""
test_demo_sequence.py
=====================
TASK-002 Definition-of-Done tests.

Run with:
    pytest perception/tests/test_demo_sequence.py -v

These tests require data/demo_sequence/ to be populated.
If it isn't, run:
    python perception/scripts/fetch_demo_sequence.py --synthetic
to generate placeholder data that satisfies the structural tests.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR   = ROOT / "data" / "demo_sequence"
VEL_DIR    = DEMO_DIR / "velodyne"
LABEL_DIR  = DEMO_DIR / "labels"

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _bin_files() -> list[Path]:
    return sorted(VEL_DIR.glob("*.bin"))

def _label_files() -> list[Path]:
    return sorted(LABEL_DIR.glob("*.label"))


@pytest.fixture(scope="session", autouse=True)
def require_demo_sequence():
    """Skip all tests in this module if the demo sequence doesn't exist."""
    if not VEL_DIR.exists() or not any(VEL_DIR.glob("*.bin")):
        pytest.skip(
            "Demo sequence not found. Run:\n"
            "  python perception/scripts/fetch_demo_sequence.py --synthetic\n"
            "to generate placeholder data, then re-run the tests."
        )


# ---------------------------------------------------------------------------
# TASK-002 Definition-of-Done tests
# ---------------------------------------------------------------------------

class TestFrameLabelCountMatch:
    """Every .bin frame must have a matching .label file, and counts must match."""

    def test_frame_count_is_nonzero(self):
        bins = _bin_files()
        assert len(bins) > 0, "No .bin frames found in data/demo_sequence/velodyne/"

    def test_label_count_is_nonzero(self):
        labels = _label_files()
        assert len(labels) > 0, "No .label files found in data/demo_sequence/labels/"

    def test_frame_count_matches_label_count(self):
        bins   = _bin_files()
        labels = _label_files()
        assert len(bins) == len(labels), (
            f"Frame/label count mismatch: {len(bins)} .bin vs {len(labels)} .label"
        )

    def test_every_bin_has_a_matching_label(self):
        """Stem-by-stem: each 000000.bin must have a 000000.label, etc."""
        bin_stems   = {f.stem for f in _bin_files()}
        label_stems = {f.stem for f in _label_files()}
        missing_labels = bin_stems - label_stems
        extra_labels   = label_stems - bin_stems
        assert not missing_labels, f"Frames without labels: {sorted(missing_labels)}"
        assert not extra_labels,   f"Labels without frames: {sorted(extra_labels)}"


class TestBinFileFormat:
    """Sanity-check the .bin binary format: float32 x,y,z,intensity columns."""

    def test_bin_is_divisible_by_16_bytes(self):
        """Each point is 4×float32 = 16 bytes; file must be a multiple of 16."""
        for f in _bin_files()[:10]:   # check first 10 frames
            size = f.stat().st_size
            assert size % 16 == 0, (
                f"{f.name}: size {size} is not a multiple of 16 bytes"
            )

    def test_bin_has_reasonable_point_count(self):
        """Velodyne-64 typically yields 50k–200k points per scan."""
        for f in _bin_files()[:10]:
            n_points = f.stat().st_size // 16
            assert 1_000 <= n_points <= 300_000, (
                f"{f.name}: suspicious point count {n_points}"
            )

    def test_bin_loads_as_float32_array(self):
        f = _bin_files()[0]
        pts = np.frombuffer(f.read_bytes(), dtype=np.float32).reshape(-1, 4)
        assert pts.ndim == 2
        assert pts.shape[1] == 4


class TestLabelFileFormat:
    """Sanity-check the .label format: uint32 per point, lower 16 bits = class."""

    def test_label_point_count_matches_bin_point_count(self):
        """Each .label file must have the same point count as its .bin partner."""
        for bin_f in _bin_files()[:10]:
            lbl_f = LABEL_DIR / bin_f.with_suffix(".label").name
            n_points_bin   = bin_f.stat().st_size // 16
            n_points_label = lbl_f.stat().st_size // 4    # uint32 = 4 bytes
            assert n_points_bin == n_points_label, (
                f"{bin_f.name}: {n_points_bin} pts in .bin vs "
                f"{n_points_label} pts in .label"
            )

    def test_label_semantic_ids_are_reasonable(self):
        """SemanticKITTI class IDs 0–259; lower 16 bits of each uint32."""
        lbl_f = _label_files()[0]
        raw = np.frombuffer(lbl_f.read_bytes(), dtype=np.uint32)
        semantic_ids = raw & 0xFFFF
        assert semantic_ids.max() < 260, (
            f"Unexpected semantic ID {semantic_ids.max()} — corrupt label file?"
        )


class TestHeroMomentClasses:
    """
    The demo sequence must contain at least one pedestrian and one vehicle —
    the 'hero moment' requirement from TASK-002.

    Note: synthetic data generated by fetch_demo_sequence.py --synthetic is
    designed to satisfy these requirements.
    """

    PERSON_ID  = 30   # SemanticKITTI: person
    CAR_ID     = 10   # SemanticKITTI: car

    def _all_semantic_ids(self) -> np.ndarray:
        ids = []
        for lbl_f in _label_files():
            raw = np.frombuffer(lbl_f.read_bytes(), dtype=np.uint32)
            ids.append(raw & 0xFFFF)
        return np.concatenate(ids) if ids else np.array([], dtype=np.uint32)

    def test_sequence_contains_pedestrian(self):
        all_ids = self._all_semantic_ids()
        assert self.PERSON_ID in all_ids, (
            f"No pedestrian (class {self.PERSON_ID}) found in the demo sequence. "
            "Pick a different frame range or sequence."
        )

    def test_sequence_contains_vehicle(self):
        all_ids = self._all_semantic_ids()
        assert self.CAR_ID in all_ids, (
            f"No vehicle (class {self.CAR_ID}) found in the demo sequence. "
            "Pick a different frame range or sequence."
        )
