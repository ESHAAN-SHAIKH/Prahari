"""Central configuration for the PRAHARI prototype.

Every geometric constant the problem statement names lives here, once, so the grid engine,
the codec and the dashboard cannot drift apart.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- geometry ---
# The quadtree root covers a square centred on the sensor. Root size is chosen so that
# MAX_DEPTH subdivisions land exactly on the 5 cm cell the PS asks for:
#     204.8 m / 2**12 = 0.05 m
ROOT_SIZE_M = 204.8
MAX_DEPTH = 12
MIN_CELL_M = ROOT_SIZE_M / (2 ** MAX_DEPTH)      # 0.05 m
MAX_RANGE_M = 100.0                              # PS: map out to 100 m

# PS resolution schedule: 5 cm inside 10 m, coarsening to ~50 cm at 100 m.
# Quadtree cell sizes are powers of two of MIN_CELL_M, so the achievable sizes are
# 0.05, 0.10, 0.20, 0.40, 0.80 ... The engine picks the finest level whose size is <=
# the target for that range; at 100 m that is 0.40 m. The level table is published to
# the dashboard rather than rounded off in a caption.
FOVEA_RADIUS_M = 10.0                            # full resolution inside this radius
COARSE_SIZE_M = 0.50                             # target at MAX_RANGE_M

# Risk-adaptive overrides. These are what make the grid *risk*-adaptive rather than merely
# range-adaptive, which is the core novelty over a plain foveated grid.
HAZARD_REFINE_FACTOR = 4.0      # hazard cells get a target 4x finer than range alone asks
LOWCONF_REFINE_FACTOR = 2.0     # degraded sensing gets 2x finer, so it is inspected not hidden
CONF_THRESHOLD = 0.60           # below this, sensing is treated as degraded
BOUNDARY_MIN_SIZE_M = 0.10      # split cells straddling a class boundary, down to 10 cm

# ------------------------------------------------------------------ classes ---
# id: (name, group, hazard)
CLASSES = {
    0:  ("unknown",        "unknown",  False),
    1:  ("drivable",       "terrain",  False),
    2:  ("rough-ground",   "terrain",  False),
    3:  ("vegetation",     "terrain",  False),
    4:  ("wall",           "static",   False),
    5:  ("pole",           "static",   False),
    6:  ("barrier",        "static",   False),
    7:  ("pedestrian",     "dynamic",  False),
    8:  ("vehicle",        "dynamic",  False),
    9:  ("two-wheeler",    "dynamic",  False),
    40: ("pothole",        "hazard",   True),
    41: ("ditch",          "hazard",   True),
    42: ("loose-rock",     "hazard",   True),
    43: ("water-crossing", "hazard",   True),
    44: ("snow-drift",     "hazard",   True),
}
HAZARD_IDS = {k for k, v in CLASSES.items() if v[2]}
CLASS_NAMES = {k: v[0] for k, v in CLASSES.items()}
CLASS_GROUPS = {k: v[1] for k, v in CLASSES.items()}

# The four categories the problem statement names, and how model classes fold into them.
PS_CATEGORIES = {
    "walls":       [4, 6],
    "poles":       [5],
    "pedestrians": [7, 9],
    "vehicles":    [8],
}

# --------------------------------------------------------------- subdivision ---
DRIVER_RANGE, DRIVER_HAZARD, DRIVER_CONFIDENCE, DRIVER_BOUNDARY = 0, 1, 2, 3
DRIVER_NAMES = {
    DRIVER_RANGE: "range",
    DRIVER_HAZARD: "hazard",
    DRIVER_CONFIDENCE: "low confidence",
    DRIVER_BOUNDARY: "class boundary",
}

# ---------------------------------------------------------------- sensor ------
LIDAR_CHANNELS = 64
LIDAR_AZIMUTH_BINS = 1024
LIDAR_FOV_UP_DEG = 2.0
LIDAR_FOV_DOWN_DEG = -24.8
SENSOR_HEIGHT_M = 1.73

# ------------------------------------------------------------------ runtime ---
TARGET_HZ = float(os.environ.get("PRAHARI_TARGET_HZ", "10"))
BYTES_PER_CELL = 6          # z:int16, class:u8, conf:u8, level:u8, driver:u8
QUANT_M = MIN_CELL_M / 2.0  # cell centres land on half-cell steps at every level
BYTES_PER_POINT = 16        # x,y,z,intensity as float32 — the raw-cloud comparison

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"

# Where the training notebook writes its evidence export. Point this at your own copy.
EVIDENCE_PATH = Path(os.environ.get(
    "PRAHARI_EVIDENCE", str(REPO_ROOT / "evidence" / "trinetra_evidence_export.json")))

# Optional real checkpoint. When unset, the perception stage runs the synthetic sensor
# model and every frame is labelled as such, end to end, in the UI.
CHECKPOINT_PATH = os.environ.get("PRAHARI_CKPT", "")
CHECKPOINT_CONFIG = os.environ.get("PRAHARI_CFG", "")


def target_cell_size(range_m: float) -> float:
    """PS resolution schedule: flat inside the fovea, linear coarsening beyond it."""
    if range_m <= FOVEA_RADIUS_M:
        return MIN_CELL_M
    t = (range_m - FOVEA_RADIUS_M) / (MAX_RANGE_M - FOVEA_RADIUS_M)
    return MIN_CELL_M + t * (COARSE_SIZE_M - MIN_CELL_M)


def level_table() -> list[dict]:
    """Level -> cell size -> the range band that first requests it. Shown in the UI so the
    schedule is inspectable rather than asserted."""
    rows = []
    for lvl in range(MAX_DEPTH + 1):
        size = ROOT_SIZE_M / (2 ** lvl)
        if size > COARSE_SIZE_M * 4 or size < MIN_CELL_M:
            continue
        lo = hi = None
        for r in range(0, int(MAX_RANGE_M) + 1):
            finest = _finest_level_for(target_cell_size(float(r)))
            if finest == lvl:
                lo = r if lo is None else lo
                hi = r
        rows.append({"level": lvl, "size_m": round(size, 4),
                     "range_from_m": lo, "range_to_m": hi})
    return rows


def _finest_level_for(target_m: float) -> int:
    """Finest quadtree level whose cell is no larger than target_m."""
    lvl = 0
    size = ROOT_SIZE_M
    while size > target_m and lvl < MAX_DEPTH:
        size /= 2.0
        lvl += 1
    return lvl
