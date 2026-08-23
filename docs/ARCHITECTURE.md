# PRAHARI-Lite — Architecture Overview

## Pipeline Summary

The system is a three-stage pipeline:

```
[Perception]  →  [Grid Engine]  →  [Backend + Dashboard]
```

Each stage is a separately owned track; they communicate through two frozen interfaces
defined in Milestone 1 so all three can develop in parallel after TASK-003.

---

## Stage 1 — Perception

**Directory:** `perception/`

**Inputs:** Raw LiDAR frames (`.bin` point clouds) + label files (`.label`) from
`data/demo_sequence/`.

**Output:** `(points: np.ndarray[N, 4], class_ids: np.ndarray[N])` per frame, where
the four point columns are `[x, y, z, intensity]` and `class_ids` use the collapsed
five-class taxonomy defined in `perception/taxonomy.py`.

**Key modules:**
- `loader.py` — reads raw binary frames from disk.
- `taxonomy.py` — maps SemanticKITTI's 20+ classes down to 5 base classes.
- `backends/groundtruth_backend.py` — guaranteed fallback using dataset labels.
- `backends/model_backend.py` — optional pretrained-model path (time-boxed).
- `service.py` — single entrypoint `classify_frame(idx)`, backend-switchable via one
  config line.

**Interface contract (frozen after TASK-003 commit):**

```python
from perception.service import classify_frame

points, class_ids = classify_frame(frame_idx)
# points:    np.ndarray, shape (N, 4), dtype float32
# class_ids: np.ndarray, shape (N,),   dtype int32
```

---

## Stage 2 — Grid Engine

**Directory:** `grid_engine/`

**Inputs:** `(points, class_ids)` from Stage 1.

**Output:** A dict of `cell_key → Cell` objects, where each `Cell` carries
`height_mean`, `height_max`, `class_id` (majority-priority vote), `point_count`, and
`tier` (which resolution ring the cell belongs to).

**Key modules:**
- `ring_grid.py` — the core novel structure: distance-tiered non-uniform grid
  (5 cm → 15 cm → 50 cm cell size across three range rings).
- `uniform_grid.py` — uniform 5 cm baseline for the memory-savings comparison.
- `memory_report.py` — computes `memory_saved_pct` from real frame data.
- `serialize.py` — converts the cell dict to a flat JSON list for API transport.

**Ring-tier config (from the problem statement's own example numbers):**

| Tier | Range | Cell size |
|------|-------|-----------|
| 0 (near) | 0 – 10 m | 5 cm |
| 1 (mid) | 10 – 30 m | 15 cm |
| 2 (far) | 30 – 100 m | 50 cm |

A point exactly at a tier boundary belongs to the *inner* (finer) tier — no
ambiguity, tested explicitly in TASK-008.

---

## Stage 3 — Backend + Dashboard

**Directory:** `app/`

**Backend (`app/backend/`):** FastAPI service exposing:
- `GET /frame` — latest adaptive grid (Stage 2 output, serialized).
- `GET /frame/uniform` — latest uniform-grid baseline.
- `POST /playback/{start,stop,speed}` — playback control.
- `GET /metrics` — FPS, per-stage latency, memory-saved %, active backend.

**Dashboard (`app/frontend/`):** React (preferred) or Streamlit application that polls
the backend on a ~200–300 ms interval and renders:
- Variable-resolution grid view (cell sizes visibly differ near vs. far).
- Side-by-side comparison against the uniform-grid baseline.
- Live metrics panel with FPS color-banding and class legend.

---

## Data flow diagram

```
data/demo_sequence/
    ├── velodyne/  (*.bin)
    └── labels/   (*.label)
          │
          ▼
  perception/loader.py
  perception/taxonomy.py
  perception/service.py  ←── PERCEPTION_BACKEND flag
          │
          │  (N, 4) points + (N,) class_ids
          ▼
  grid_engine/ring_grid.py   →  dict[cell_key, Cell]
  grid_engine/serialize.py   →  JSON payload
          │
          ▼
  app/backend/playback.py    (background loop)
  app/backend/main.py        (FastAPI endpoints)
          │
          │  HTTP polling (~200–300 ms)
          ▼
  app/frontend/              (React / Streamlit dashboard)
```

---

## Scoping notes

- Physical hardware, cloud infra, and multi-day training runs are **out of scope**.
- All performance numbers are laptop-class and labeled as such.
- Edge deployment is future work, documented in TASK-021's slide deck.
