# PRAHARI-Lite: Adaptive Variable-Resolution 2.5D LiDAR Perception MVP

> Hackathon implementation for **DRDO Problem Statement 26053** —
> *Adaptive Variable Resolution 2.5D Lidar Mapping for Dynamic Environment Perception*.

---

## What it does

PRAHARI-Lite ingests a replayed public LiDAR sequence (SemanticKITTI), classifies
every point into terrain / static-obstacle / dynamic-pedestrian / dynamic-vehicle, and
projects the result into a **genuinely non-uniform 2.5D grid** (5 cm near-field →
50 cm far-field). A live dashboard shows the color-coded result side-by-side with a
uniform-grid baseline and displays real FPS / latency / memory-saved metrics.

---

## Quick-start

```bash
# 1. Create & activate the virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the backend
uvicorn app.backend.main:app --reload

# 4. (React path) Run the frontend
cd app/frontend && npm install && npm run dev

# 5. Run all tests
pytest -v
```

---

## Repository layout

```
PRAHARI/
├── perception/          # Point-cloud classification (TASK-003 to TASK-006)
│   ├── backends/        #   groundtruth_backend.py, model_backend.py
│   ├── scripts/         #   fetch_demo_sequence.py
│   ├── tests/
│   ├── loader.py
│   ├── taxonomy.py
│   └── service.py
│
├── grid_engine/         # Variable-resolution 2.5D grid (TASK-007 to TASK-011)
│   ├── tests/
│   ├── cell.py
│   ├── ring_grid.py
│   ├── uniform_grid.py
│   ├── serialize.py
│   └── memory_report.py
│
├── app/
│   ├── backend/         # FastAPI service + playback loop (TASK-012 to TASK-014)
│   │   ├── tests/
│   │   ├── main.py
│   │   ├── playback.py
│   │   └── metrics.py
│   └── frontend/        # React (or Streamlit) dashboard (TASK-015 to TASK-018)
│
├── data/                # Demo sequence lives here — NOT committed to git
│   └── README.md
│
├── notebooks/           # Exploratory analysis
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEMO_SCRIPT.md
│   └── reports/         # Generated benchmark reports
│
├── requirements.txt
└── .gitignore
```

---

## Milestones & track ownership

| Milestone | Tasks | Track |
|-----------|-------|-------|
| 1 — Foundation & Data Pipeline | TASK-001 → 003 | All (setup) |
| 2 — Perception Layer | TASK-004 → 006 | Perception |
| 3 — Variable-Resolution Grid (**core novelty**) | TASK-007 → 011 | Grid Engine |
| 4 — Backend API & Playback Loop | TASK-012 → 014 | Backend |
| 5 — Real-Time Dashboard | TASK-015 → 018 | Frontend |
| 6 — Metrics, Packaging & Demo Readiness | TASK-019 → 021 | All |

> Milestones 2 and 3 are designed to run **in parallel** after TASK-003's shared
> interface is committed.

---

## Key design decisions & scoping assumptions

- Dataset: trimmed SemanticKITTI sequence replayed from disk (no physical LiDAR).
- Edge deployment is **explicitly out of scope** for the hackathon MVP; all
  performance numbers are labeled as laptop-class.
- Ground-truth-label relabeling (TASK-004) is the guaranteed fallback for
  perception; the pretrained-model path (TASK-005) is time-boxed.
- See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the three-box pipeline overview.

---

## License

For hackathon / evaluation purposes only. Dataset usage governed by SemanticKITTI's
own licence terms.
