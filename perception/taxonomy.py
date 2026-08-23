"""
taxonomy.py
===========
Collapsed SemanticKITTI class taxonomy for PRAHARI-Lite.

SemanticKITTI uses ~20+ fine-grained classes.  For this MVP we collapse
them into 5 base classes that the grid engine, dashboard, and metrics
layer all refer to.  The mapping is done once here; no other module
defines or invents class IDs.

FROZEN AFTER TASK-003 COMMIT -- do not add new base classes without
updating every downstream consumer (grid_engine, app/backend, app/frontend).
"""

from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Base class taxonomy  (ID -> human name)
# ---------------------------------------------------------------------------

BASE_CLASSES: dict[int, str] = {
    0:   "terrain_drivable",
    1:   "static_obstacle",
    2:   "dynamic_pedestrian",
    3:   "dynamic_vehicle",
    255: "unknown",
}

# Human-readable name -> ID (convenience reverse map)
BASE_CLASS_IDS: dict[str, int] = {v: k for k, v in BASE_CLASSES.items()}

# Color palette (R, G, B) -- single source of truth used by every renderer.
# Keep these distinct and intuitive for a live demo audience.
BASE_CLASS_COLORS: dict[int, tuple] = {
    0:   (100, 180, 100),   # terrain_drivable   -- muted green
    1:   (200, 130,  60),   # static_obstacle     -- amber/orange
    2:   (220,  60,  60),   # dynamic_pedestrian  -- red
    3:   ( 60, 120, 220),   # dynamic_vehicle     -- blue
    255: (120, 120, 120),   # unknown             -- grey
}

# Priority order for majority-vote within a grid cell (TASK-007).
# Highest-priority class wins ties -- safety-critical: an obstacle vote
# must never be silently overridden by a terrain majority.
CLASS_PRIORITY: dict[int, int] = {
    2:   4,   # dynamic_pedestrian -- highest
    3:   3,   # dynamic_vehicle
    1:   2,   # static_obstacle
    0:   1,   # terrain_drivable
    255: 0,   # unknown            -- lowest
}

# ---------------------------------------------------------------------------
# Full SemanticKITTI -> BASE_CLASSES mapping
# ---------------------------------------------------------------------------
# Every SemanticKITTI semantic class ID (lower 16 bits of the uint32 label)
# must appear here.  Classes not relevant to the MVP map to 255 (unknown).
# Validate with validate_completeness() -- "unknown" silencing an obstacle
# is a real safety bug even at demo scale.
#
# Reference: http://semantic-kitti.org/dataset.html#format (label definitions)

SEMANTICKITTI_TO_BASE: dict[int, int] = {
    # ---- Unlabeled / noise ------------------------------------------------
    0:   255,  # unlabeled     -> unknown
    1:   255,  # outlier       -> unknown

    # ---- Vehicles (dynamic) -----------------------------------------------
    10:  3,    # car           -> dynamic_vehicle
    11:  3,    # bicycle       -> dynamic_vehicle  (ridden)
    13:  3,    # bus           -> dynamic_vehicle
    15:  3,    # motorcycle    -> dynamic_vehicle
    16:  3,    # on-rails      -> dynamic_vehicle
    18:  3,    # truck         -> dynamic_vehicle
    20:  3,    # other-vehicle -> dynamic_vehicle

    # ---- People (dynamic) -------------------------------------------------
    30:  2,    # person        -> dynamic_pedestrian
    31:  2,    # bicyclist     -> dynamic_pedestrian
    32:  2,    # motorcyclist  -> dynamic_pedestrian

    # ---- Ground / drivable surface ----------------------------------------
    40:  0,    # road          -> terrain_drivable
    44:  0,    # parking       -> terrain_drivable
    48:  0,    # sidewalk      -> terrain_drivable  (walkable surface)
    49:  0,    # other-ground  -> terrain_drivable
    60:  0,    # lane-marking  -> terrain_drivable

    # ---- Static structures ------------------------------------------------
    50:  1,    # building      -> static_obstacle
    51:  1,    # fence         -> static_obstacle
    52:  1,    # other-structure -> static_obstacle
    70:  1,    # vegetation    -> static_obstacle   (blocks path)
    71:  1,    # trunk         -> static_obstacle
    72:  0,    # terrain       -> terrain_drivable  (natural ground)
    80:  1,    # pole          -> static_obstacle
    81:  1,    # traffic-sign  -> static_obstacle
    99:  1,    # other-object  -> static_obstacle

    # ---- Moving instances (upper 16 bits set; lower 16 = semantic class)
    # SemanticKITTI moving-instance IDs are semantic IDs + 0xF200 prefix.
    # The loader masks to lower 16 bits, so these map through the base IDs
    # above.  The entries below are included for completeness / defensiveness.
    252: 3,    # moving-car
    253: 2,    # moving-bicyclist  -> dynamic_pedestrian
    254: 2,    # moving-person
    255: 2,    # moving-motorcyclist
    256: 3,    # moving-on-rails
    257: 3,    # moving-bus
    258: 3,    # moving-truck
    259: 3,    # moving-other-vehicle
}

# The DEFAULT for any class not explicitly listed above.
# Using 255 (unknown) is safe -- a missing mapping is surfaced, not silenced.
_DEFAULT_BASE_CLASS: int = 255

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def map_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Map a raw SemanticKITTI label array to base class IDs.

    Parameters
    ----------
    raw_labels : np.ndarray, shape (N,), dtype uint32
        Per-point raw SemanticKITTI label values (lower 16 bits = semantic
        class, upper 16 bits = instance ID -- we use only the lower 16).

    Returns
    -------
    np.ndarray, shape (N,), dtype int32
        Per-point base class IDs drawn from BASE_CLASSES.

    Notes
    -----
    Any semantic ID not present in SEMANTICKITTI_TO_BASE is silently mapped
    to 255 (unknown).  Call validate_completeness() to catch gaps before
    demo time.
    """
    # Extract semantic class (lower 16 bits); ignore instance upper bits.
    semantic_ids = (raw_labels & 0xFFFF).astype(np.int32)

    # Vectorised lookup via a lookup table (LUT).
    # Build a LUT up to max(semantic_ids)+1, defaulting to _DEFAULT_BASE_CLASS.
    max_id = int(semantic_ids.max()) if len(semantic_ids) else 0
    lut_size = max(max_id + 1, max(SEMANTICKITTI_TO_BASE.keys()) + 1)
    lut = np.full(lut_size, _DEFAULT_BASE_CLASS, dtype=np.int32)
    for raw_id, base_id in SEMANTICKITTI_TO_BASE.items():
        if raw_id < lut_size:
            lut[raw_id] = base_id

    # Clamp any out-of-range IDs to _DEFAULT_BASE_CLASS
    semantic_clipped = np.clip(semantic_ids, 0, lut_size - 1)
    return lut[semantic_clipped]


def validate_completeness(
    sequence_dir: Optional[Path] = None,
    sample_frames: int = 50,
) -> dict[int, str]:
    """Scan label files and return any unmapped raw SemanticKITTI class IDs.

    Parameters
    ----------
    sequence_dir : Path, optional
        Directory containing a 'labels/' subfolder with *.label files.
        Defaults to data/demo_sequence/ relative to the project root.
    sample_frames : int
        Maximum number of label frames to inspect (for speed).

    Returns
    -------
    dict[int, str]
        Mapping of {raw_class_id: "UNMAPPED"} for every class ID found in
        the sequence that is NOT in SEMANTICKITTI_TO_BASE.  An empty dict
        means the taxonomy is complete.
    """
    if sequence_dir is None:
        sequence_dir = Path(__file__).resolve().parents[1] / "data" / "demo_sequence"

    labels_dir = sequence_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(
            "Labels directory not found: {}.\n"
            "Run: python perception/scripts/fetch_demo_sequence.py --synthetic".format(labels_dir)
        )

    label_files = sorted(labels_dir.glob("*.label"))[:sample_frames]
    if not label_files:
        raise FileNotFoundError("No .label files found in {}".format(labels_dir))

    seen_ids: set = set()
    for lf in label_files:
        raw = np.frombuffer(lf.read_bytes(), dtype=np.uint32)
        seen_ids.update((raw & 0xFFFF).tolist())

    unmapped = {
        sid: "UNMAPPED"
        for sid in sorted(seen_ids)
        if sid not in SEMANTICKITTI_TO_BASE
    }
    return unmapped
