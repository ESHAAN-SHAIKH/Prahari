# Hackathon Engineering Blueprint — PRAHARI-Lite: Adaptive Variable-Resolution 2.5D LiDAR Perception MVP

**PRAHARI-Lite** is the hackathon-scoped reference implementation for DRDO Problem Statement 26053 — *Adaptive Variable Resolution 2.5D Lidar Mapping for Dynamic Environment Perception*. It is a deliberate, honestly-scoped subset of the full PRAHARI production blueprint: it ingests recorded LiDAR frames, classifies terrain/static/dynamic points, projects the result into a genuinely non-uniform 2.5D grid (fine near the sensor, coarse far away), and renders it live on a dashboard with real FPS/latency/memory metrics. Everything cut from the full production system is cut *explicitly*, not silently, so a judging panel sees engineering judgment rather than missing scope.

## PROJECT DESCRIPTION

**Product:** A demoable software pipeline that turns a replayed public LiDAR sequence into a live, color-coded, variable-resolution elevation-and-semantic map, with a side-by-side comparison against a uniform high-resolution grid to make the memory-savings claim visible, not just asserted.

**Users (for this MVP):** the hackathon judging panel, watching a live or recorded demo — there is no real onboard vehicle, remote operator fleet, or mission-analysis persona at this scope; those are documented as post-hackathon extensions.

**Core features (MVP-scoped):**
1. Point classification into 3–5 classes (terrain / static obstacle / dynamic pedestrian / dynamic vehicle), via a fallback ground-truth relabeling path and, time permitting, a real pretrained model.
2. Variable-resolution 2.5D grid engine: distance-tiered subdivision (5cm near-field → 50cm far-field), with a stretch goal of one risk-adaptive rule.
3. Live dashboard: color-coded grid render, uniform-vs-adaptive side-by-side comparison, FPS/latency/memory-saved metrics panel.
4. A reproducible benchmark run over a fixed demo sequence, with headline numbers a judge can be shown and trust.

**Technical requirements & constraints (carried over from the full problem statement, MVP-scoped):**
- Grid projection must not lose or double-count points, and must not have ambiguous behavior at tier boundaries — this is tested explicitly, not just assumed.
- The system must visibly demonstrate memory reduction versus a uniform high-resolution grid over the same range, using real computed numbers from the actual demo sequence.
- Latency/FPS must be genuinely measured on whatever hardware is used, labeled honestly (laptop CPU/GPU, not a fabricated edge-device number).
- No dependency on physical hardware, cloud infrastructure, or multi-day training runs — everything must run on a laptop within the hackathon window.

**Assumptions (explicitly labeled, since this is a compressed scope, not the original problem statement):**
- **[ASSUMPTION]** Team size is 3–4 people, enabling two to three parallel work tracks (Perception, Grid Engine, Backend+Dashboard) rather than the strictly sequential dependency chain a solo developer would need.
- **[ASSUMPTION]** Total build window is ~30–36 hours wall-clock (typical overnight-to-48h hackathon format), with the task estimates below sized for a small parallel team, not a solo sequential build.
- **[ASSUMPTION]** Data source is a trimmed SemanticKITTI (or KITTI + labels) sequence, replayed from disk to simulate a live sensor feed — no physical LiDAR is used.
- **[ASSUMPTION]** "Edge deployment" claims are explicitly out of scope for this MVP; all benchmark numbers are reported as laptop-class numbers, clearly labeled, with edge deployment named as future work.
- **[ASSUMPTION]** Ground-truth-label replay is an accepted, disclosed stand-in for live model inference if the model integration proves too time-risky — this is treated as a legitimate scoping decision, not a shortcut to hide.

---

## Phase 1: MVP Development Roadmap

### How to use this roadmap

- Tasks are numbered continuously (TASK-001 → TASK-021) and grouped into 6 milestones. Unlike the full production blueprint, several tasks are designed to run in **parallel tracks** across a small team — dependency notes call this out explicitly.
- Every task lists **Prerequisites**, and a **Definition of Done** checklist — treat these the same way the full blueprint does: don't mark a box checked until it's actually been run.
- **Estimated Time** assumes a small team working in parallel, each member reasonably comfortable with their track's tools (Python/NumPy, FastAPI, basic frontend). Add buffer if any track requires learning a new tool from scratch.
- 🔬 marks the tasks that carry the actual novelty a judge should walk away remembering. Protect these tasks' time budget first if the schedule slips.
- ⭐ marks stretch tasks — attempt only after every non-stretch task's Definition of Done is fully checked.

**Total milestones:** 6
**Total estimated build time:** ~38 hours of task effort (~24–30 hours realistic wall-clock with a 3–4 person team running Perception, Grid Engine, and Backend/Dashboard tracks partly in parallel)
**Assumed team availability:** a single hackathon sprint (overnight-to-48h format), not spread across weeks

---

## MILESTONE 1: Foundation & Data Pipeline

**Goal:** A working repo skeleton and a trimmed, loop-ready LiDAR sequence with labels, so every downstream track has real data to build against from hour 1.

**Estimated time:** ~4 hours (TASK-001 → TASK-003)

### TASK-001: Initialize Hackathon Repo & Environment

**Objective:** Stand up a minimal monorepo — no CI, no Docker orchestration, just a clean structure everyone can clone and run.

**Prerequisites:** Git, Python 3.10+, Node.js (only if doing the React dashboard track).

**Estimated Difficulty:** Easy
**Estimated Time:** 1 hour

## Install:
```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy fastapi uvicorn pydantic
```

## Create:
```bash
mkdir -p perception grid_engine app/backend app/frontend data notebooks docs
git init
```

## Files to Create:
```text
/README.md
/requirements.txt
/.gitignore
/docs/ARCHITECTURE.md
```

## Configuration:
```text
# .gitignore — critical: don't let anyone accidentally commit the dataset
.venv/
__pycache__/
node_modules/
data/*.bin
data/*.label
*.pyc
```

## Implementation Steps:
1. Create the folder structure exactly as shown — this is referenced by every later task, don't rename mid-hackathon.
2. Write a one-paragraph `docs/ARCHITECTURE.md`: perception → grid_engine → backend → frontend, matching the 3-box diagram in the pitch deck.
3. Assign the three tracks now (Perception, Grid Engine, Backend+Dashboard) so people aren't blocked waiting on each other past hour 4.
4. Commit as `chore: init PRAHARI-Lite skeleton`.

## Tests Required:
- N/A (scaffolding).

## Verification:
```bash
git log --oneline -1
ls perception grid_engine app/backend app/frontend
```

## Definition of Done:
```text
[ ] Folder structure matches spec
[ ] requirements.txt and .gitignore committed
[ ] docs/ARCHITECTURE.md written
[ ] Tracks assigned, everyone unblocked
```

## Common Mistakes:
- Spending the first 2 hours picking a "proper" microservice/Docker setup — for a 30-hour build this is pure overhead; one repo, one venv, one process is correct here.

---

### TASK-002: Acquire & Trim a Demo LiDAR Sequence

**Objective:** Get one good SemanticKITTI (or KITTI+labels) sequence, trimmed to a length that loops cleanly and contains a genuinely interesting mix of terrain, static obstacles, and moving objects.

**Prerequisites:** TASK-001.

**Estimated Difficulty:** Easy
**Estimated Time:** 1.5 hours

## Install:
```bash
pip install requests tqdm
```

## Files to Create:
```text
/perception/scripts/fetch_demo_sequence.py
/data/README.md
```

## Configuration:
```python
# fetch_demo_sequence.py (excerpt)
# [ASSUMPTION] Using a single short SemanticKITTI sequence (e.g., a ~300-500 frame
# slice of sequence 00 or 04) chosen for a clear mix of road, poles, pedestrians, cars —
# NOT the full dataset. Trimming early avoids playback loops that drag during the demo.
TARGET_SEQUENCE = "04"
FRAME_RANGE = (0, 400)
```

## Implementation Steps:
1. Download the chosen sequence's `velodyne/` (`.bin` point clouds) and `labels/` (`.label` files) per SemanticKITTI's published layout.
2. Slice to the `FRAME_RANGE` window — pick a segment by eyeballing a few frames first, favoring one with a pedestrian or vehicle passing near the sensor (this becomes your demo's "hero moment").
3. Copy the trimmed slice into `data/demo_sequence/{velodyne,labels}/`.
4. Write `data/README.md`: which sequence, which frame range, why it was chosen, and the exact source URL for reproducibility.

## Tests Required:
- `test_frame_count_matches_label_count`: every `.bin` frame has a matching `.label` file, and counts line up.

## Verification:
```bash
python perception/scripts/fetch_demo_sequence.py
ls data/demo_sequence/velodyne | wc -l
ls data/demo_sequence/labels | wc -l
pytest perception/tests/test_demo_sequence.py -v
```

## Definition of Done:
```text
[ ] Trimmed sequence downloaded and verified frame/label count match
[ ] Sequence contains at least one clear pedestrian and one clear vehicle pass
[ ] data/README.md documents source and selection rationale
```

## Common Mistakes:
- Downloading the full multi-GB dataset instead of one sequence — wastes hours of hackathon time on disk I/O and bandwidth for data you'll never use.
- Picking a segment that's entirely empty road — visually boring and won't show off object classification during the demo.

---

### TASK-003: Unified Point Loader & Class Taxonomy

**Objective:** One shared function every track calls to get `(points, classes)` for a frame — this is the interface contract that lets Perception, Grid Engine, and Backend tracks work in parallel without blocking on each other.

**Prerequisites:** TASK-002.

**Estimated Difficulty:** Easy
**Estimated Time:** 1.5 hours

## Files to Create:
```text
/perception/loader.py
/perception/taxonomy.py
```

## Configuration:
```python
# perception/taxonomy.py
# Collapsed from SemanticKITTI's ~20+ classes down to what the demo actually needs.
BASE_CLASSES = {
    "terrain_drivable": 0,
    "static_obstacle": 1,
    "dynamic_pedestrian": 2,
    "dynamic_vehicle": 3,
    "unknown": 255,
}
# raw SemanticKITTI label id -> BASE_CLASSES id, filled in during implementation
SEMANTICKITTI_TO_BASE = {
    40: 0,   # road -> terrain_drivable
    44: 0,   # parking -> terrain_drivable
    50: 1,   # building -> static_obstacle
    71: 1,   # pole -> static_obstacle
    30: 2,   # person -> dynamic_pedestrian
    10: 3,   # car -> dynamic_vehicle
    # ... remaining classes mapped or routed to "unknown" — see validate_completeness()
}
```

## Implementation Steps:
1. Implement `loader.py`: `load_frame(idx) -> (points: np.ndarray[N,4], raw_labels: np.ndarray[N])` reading directly from `data/demo_sequence/`.
2. Implement `taxonomy.py`: `map_labels(raw_labels) -> base_class_ids` plus a `validate_completeness()` that lists any raw SemanticKITTI class present in the sequence with no mapping entry — run this once and fill gaps now, not mid-demo-prep.
3. Expose the single combined entrypoint: `get_classified_points(frame_idx) -> (points, base_class_ids)` — this is the function every other track imports; freeze its signature now.
4. Commit — this file is now load-bearing for the rest of the hackathon; don't casually change its return shape later.

## Tests Required:
- `test_loader_shapes`: points are `(N,4)`, labels are `(N,)`, same `N`.
- `test_taxonomy_completeness`: no raw class in the trimmed sequence maps to nothing (silently dropped classes = invisible obstacles later).

## Verification:
```bash
pytest perception/tests/test_loader.py perception/tests/test_taxonomy.py -v
python -c "from perception.loader import get_classified_points; p,c = get_classified_points(0); print(p.shape, c.shape)"
```

## Definition of Done:
```text
[ ] get_classified_points(idx) implemented and frozen as the shared interface
[ ] Taxonomy covers every raw class present in the trimmed sequence
[ ] Tests pass
```

## Common Mistakes:
- Letting each track write its own ad hoc loader "for now" — guarantees an integration-hour disaster later when the grid engine expects a different array shape than perception actually produces. One loader, one contract, used by everyone.

---

## MILESTONE 2: Perception Layer

**Goal:** A working `classify(frame_idx)` step, with a real-model path attempted but a ground-truth-relabeling fallback guaranteed to work, so the rest of the pipeline is never blocked on model integration risk.

**Estimated time:** ~6 hours (TASK-004 → TASK-006) — **can run in parallel with Milestone 3 (Grid Engine)**, since both only depend on TASK-003's interface, not on each other.

### TASK-004: Ground-Truth Relabeling Path (Fallback — Build This First)

**Objective:** Get a guaranteed-working "perception" signal flowing end-to-end today, using the dataset's own ground-truth labels remapped through TASK-003's taxonomy — this de-risks every downstream track immediately.

**Prerequisites:** TASK-003.

**Estimated Difficulty:** Easy
**Estimated Time:** 2 hours

## Files to Create:
```text
/perception/backends/groundtruth_backend.py
```

## Implementation Steps:
1. Implement `groundtruth_backend.py`: `classify(frame_idx) -> (points, class_ids)`, literally calling `get_classified_points` from TASK-003 — this "backend" is intentionally trivial.
2. Wrap with a small artificial per-frame delay (a few ms) so downstream latency measurements aren't reporting a suspiciously-instant 0ms "inference" time that would look fabricated in the metrics panel — label this delay explicitly in a code comment as a placeholder for real inference time.
3. Confirm this backend alone is sufficient to drive the grid engine, backend API, and dashboard end-to-end — this is your safety net for the rest of the hackathon.

## Tests Required:
- `test_groundtruth_backend_matches_loader_output`: sanity check it's not silently transforming the data.

## Verification:
```bash
pytest perception/tests/test_groundtruth_backend.py -v
```

## Definition of Done:
```text
[ ] Ground-truth backend implemented and returns correct shapes
[ ] Confirmed sufficient to drive the rest of the pipeline (smoke-tested against grid engine once TASK-007 exists)
```

## Common Mistakes:
- Treating this as "not real work" and deprioritizing it — this task is what guarantees you have a demoable pipeline no matter what happens with TASK-005. Build it first, not last.

---

### TASK-005: Pretrained Model Inference Path (Primary — Time-Boxed)

**Objective:** Attempt a real pretrained point-cloud segmentation model in the loop, **strictly time-boxed** so a struggling integration doesn't eat time needed elsewhere.

**Prerequisites:** TASK-003. Independent of TASK-004 (can be built in parallel by a second person), but TASK-004 must exist as the fallback before this task's time-box expires.

**Estimated Difficulty:** Hard
**Estimated Time:** 3 hours (hard cap — see Common Mistakes)

## Install:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only is fine for a demo
# plus whichever lightweight pretrained point-cloud segmentation checkpoint is available
```

## Files to Create:
```text
/perception/backends/model_backend.py
```

## Implementation Steps:
1. Load a small pretrained point-cloud segmentation checkpoint (e.g., a lightweight PointNet++-style or similar model with an available public checkpoint trained on a KITTI-like class set).
2. Implement `model_backend.py`: `classify(frame_idx) -> (points, class_ids)`, running real inference on the loaded points, then mapping the model's native class output through a small model-specific-to-`BASE_CLASSES` table (reuse the taxonomy pattern from TASK-003, don't invent a second mapping mechanism).
3. Benchmark inference time per frame on the actual demo laptop — record it now, you'll need it for TASK-014's metrics.
4. **Hard time-box:** if this isn't producing sane-looking output by the 3-hour mark, stop, commit whatever exists as an experimental branch, and switch the pipeline to TASK-004's fallback for the rest of the hackathon. This is a planned decision, not a failure.

## Tests Required:
- `test_model_backend_output_shape`: matches TASK-004's interface exactly.
- `test_model_backend_runs_within_time_box`: informal — just track wall-clock against the 3-hour cap.

## Verification:
```bash
pytest perception/tests/test_model_backend.py -v
python -c "from perception.backends.model_backend import classify; p,c = classify(0); print(p.shape, c.shape)"
```

## Definition of Done:
```text
[ ] EITHER: model backend implemented, tested, and integrated as primary
[ ] OR: explicitly abandoned at the time-box with fallback confirmed active — both are acceptable outcomes
[ ] Whichever path is live, it matches TASK-004's exact interface
```

## Common Mistakes:
- Letting "get the real model working" become an open-ended time sink — a hackathon is won on a working end-to-end demo with an honestly-labeled fallback, not on a half-integrated model that crashes during judging. Respect the time-box.

---

### TASK-006: Perception Service Wrapper & Backend Switch

**Objective:** One entrypoint the rest of the system calls, with a simple flag to select ground-truth vs. model backend — so the demo can switch instantly if something breaks live.

**Prerequisites:** TASK-004, TASK-005 (whichever outcome).

**Estimated Difficulty:** Easy
**Estimated Time:** 1 hour

## Files to Create:
```text
/perception/service.py
```

## Configuration:
```python
# perception/service.py
PERCEPTION_BACKEND = "groundtruth"   # "groundtruth" | "model" — flip this, nothing else, to switch
```

## Implementation Steps:
1. Implement `classify_frame(idx) -> (points, class_ids)`, dispatching to whichever backend `PERCEPTION_BACKEND` selects.
2. Log which backend is active on startup — you want this visible in terminal output during the live demo so you (and judges, if asked) always know which path is running.
3. Confirm both backends are swappable without touching any downstream code — this is the whole point of TASK-003's frozen interface.

## Tests Required:
- `test_service_dispatches_to_correct_backend`.

## Verification:
```bash
pytest perception/tests/test_service.py -v
```

## Definition of Done:
```text
[ ] Single entrypoint implemented, backend-switchable via one config line
[ ] Active backend logged clearly on startup
[ ] Downstream code (once it exists) never needs to know which backend is active
```

## Common Mistakes:
- Hardcoding the model backend as the only path right before the demo "because it worked in testing" — keep the switch live and tested until the moment you're walking on stage.

---

## MILESTONE 3: Variable-Resolution Grid Engine (Core Novelty)

**Goal:** The actual thing the problem statement is asking for — a genuinely non-uniform 2.5D grid, with a real, tested guarantee against alignment errors and data loss, plus the memory-savings number that becomes your headline metric.

**Estimated time:** ~9 hours (TASK-007 → TASK-011) — **can run in parallel with Milestone 2**, since it only needs *some* `(points, class_ids)` source, which TASK-004's fallback provides from hour ~6 onward.

### TASK-007: 🔬 Ring-Tier Variable-Resolution Grid

**Objective:** Build the core novel data structure — cell size grows with distance from the sensor, implemented correctly and simply enough to finish today.

**Prerequisites:** TASK-003 (interface contract; doesn't need TASK-004/005 to actually be finished, can develop against synthetic points first).

**Estimated Difficulty:** Medium
**Estimated Time:** 3 hours

## Install:
```bash
pip install numpy
```

## Files to Create:
```text
/grid_engine/cell.py
/grid_engine/ring_grid.py
```

## Configuration:
```python
# ring_grid.py — tier config, matches the problem statement's own example numbers
RING_TIERS = [
    {"max_range_m": 10.0, "cell_size_m": 0.05},   # 5cm near-field
    {"max_range_m": 30.0, "cell_size_m": 0.15},
    {"max_range_m": 100.0, "cell_size_m": 0.50},  # 50cm far-field
]
```

## Implementation Steps:
1. Implement `Cell`: `height_mean`, `height_max`, `class_id`, `point_count`, `tier`.
2. Implement `RingGrid.project(points, class_ids) -> dict[cell_key, Cell]`: for each point, compute `r = sqrt(x²+y²)`, select the containing tier, compute the cell index within that tier's own local grid, update running stats (majority-class vote, with a fixed priority order — hazard/obstacle classes should win ties over terrain, since silently letting terrain override an obstacle vote is a real safety bug even in a demo).
3. **Boundary rule, explicit and tested:** a point exactly at a tier's `max_range_m` belongs to the *inner* (finer) tier — no ambiguity, no double-assignment.
4. Vectorize with NumPy (bin by tier first via boolean masks, then vectorized cell-index computation per tier) — a naive per-point Python loop over ~50k-120k points/frame will not hit a usable FPS; this is the single most important performance decision in the grid engine.

## Logic / Business Rules:
- Class priority for majority vote within a cell (highest wins): `dynamic_pedestrian` > `dynamic_vehicle` > `static_obstacle` > `terrain_drivable` > `unknown`. This means a cell containing even a few obstacle/dynamic points never gets silently smoothed into "terrain" by a numeric majority — directly relevant to the problem statement's safety framing.

## Tests Required:
- `test_tier_assignment_by_range`: known-range synthetic points land in the expected tier.
- `test_boundary_point_goes_to_inner_tier`: a point at exactly 10.0m lands in the 5cm tier.
- `test_class_priority_vote`: a cell with mixed points reports the highest-priority class present, not a raw plurality.

## Verification:
```bash
pytest grid_engine/tests/test_ring_grid.py -v
python -c "
from grid_engine.ring_grid import RingGrid
import numpy as np, time
pts = np.random.uniform(-60,60,(100000,3))
t0=time.time(); RingGrid().project(pts, np.zeros(100000,dtype=int)); print('projection time:', time.time()-t0)
"
```

## Definition of Done:
```text
[ ] RingGrid implemented, vectorized (not a per-point Python loop)
[ ] Boundary rule implemented and tested
[ ] Class-priority majority vote implemented and tested
[ ] Projection of a 100k-point synthetic frame completes well within your FPS budget (check against TASK-014's target)
```

## Common Mistakes:
- Writing the natural, readable per-point `for point in points:` loop first and never coming back to vectorize it — this is the #1 way a grid engine that works perfectly in a unit test fails to hit a demoable FPS on stage.

---

### TASK-008: 🔬 Alignment & Point-Conservation Test Suite

**Objective:** Prove, not just claim, the problem statement's explicit requirement — "without causing alignment errors or data loss" — with tests you can point to on request.

**Prerequisites:** TASK-007.

**Estimated Difficulty:** Easy
**Estimated Time:** 1.5 hours

## Files to Create:
```text
/grid_engine/tests/test_conservation.py
```

## Implementation Steps:
1. `test_point_count_conserved`: total point count across all cells in the output equals input point count, exactly, for both synthetic and real demo-sequence frames.
2. `test_every_point_within_its_cell_bounds`: for a sample of points, confirm the assigned cell's bounding box actually contains that point's `(x,y)` — catches off-by-one indexing bugs in the tier-local grid math.
3. `test_no_point_assigned_to_two_cells`: assert the projection is a proper partition, not an overlapping assignment.
4. Run all three against every frame in the trimmed demo sequence once (not just a handful of synthetic cases) — this is the evidence you'd actually show a skeptical judge.

## Tests Required:
- (This task *is* the test suite — see Implementation Steps.)

## Verification:
```bash
pytest grid_engine/tests/test_conservation.py -v
python grid_engine/scripts/run_conservation_check_full_sequence.py
```

## Definition of Done:
```text
[ ] All three conservation/alignment tests implemented
[ ] Tests pass against every frame in the trimmed demo sequence, not just synthetic fixtures
```

## Common Mistakes:
- Only testing against small hand-built synthetic scenes — real KITTI point distributions (dense near-field, sparse far-field, occasional outlier returns) can surface edge cases synthetic tests miss. Always also run the full real sequence through this check at least once.

---

### TASK-009: 🔬 Memory-Footprint Comparator (Uniform vs. Adaptive)

**Objective:** Compute the headline memory-savings number from real data, not an estimate — this is one of the two numbers judges will remember.

**Prerequisites:** TASK-007.

**Estimated Difficulty:** Easy
**Estimated Time:** 1.5 hours

## Files to Create:
```text
/grid_engine/uniform_grid.py
/grid_engine/memory_report.py
```

## Implementation Steps:
1. Implement `uniform_grid.py`: the same projection logic as `RingGrid`, but with a single fixed 5cm cell size out to the full 100m radius — this is the "naive" baseline the problem statement explicitly asks you to compare against.
2. Implement `memory_report.py`: for a given frame (or averaged across the full demo sequence), compute `cell_count_uniform`, `cell_count_adaptive`, and `memory_saved_pct = (1 - cell_count_adaptive/cell_count_uniform) * 100`. Also compute actual byte sizes (cell struct size × count) for a more concrete "MB vs MB" framing, not just a percentage.
3. Run across the full trimmed sequence, save a small `docs/reports/MEMORY_COMPARISON.md` with the resulting numbers — this is what goes on your metrics slide.

## Tests Required:
- `test_uniform_grid_cell_count_matches_naive_formula`: sanity-check against `(2×100/0.05)²` as an upper bound.
- `test_adaptive_always_smaller_or_equal`: the adaptive grid's cell count must never exceed the uniform grid's on any frame — if this fails, something is wrong with the tier config, not just an unlucky frame.

## Verification:
```bash
pytest grid_engine/tests/test_memory_report.py -v
python grid_engine/memory_report.py --sequence data/demo_sequence --all-frames
```

## Definition of Done:
```text
[ ] Uniform-grid baseline implemented
[ ] Memory comparison computed from real demo-sequence frames, not estimated
[ ] docs/reports/MEMORY_COMPARISON.md generated with the final headline number
[ ] Adaptive-always-smaller-or-equal invariant tested and holds on every frame
```

## Common Mistakes:
- Reporting only a best-case single frame's savings — compute and report the number across the whole demo sequence (average, plus min/max) so it can't be dismissed as cherry-picked.

---

### TASK-010: ⭐ [Stretch] Risk-Adaptive Subdivision Rule

**Objective:** One small rule that subdivides a cell finer than its range alone would justify, if local height-variance is high — this is what upgrades the story from "distance-adaptive" to genuinely "risk-adaptive," and it's a small, contained change if the core grid (TASK-007) is solid.

**Prerequisites:** TASK-007, TASK-008 passing. **Do not start this until the non-stretch Milestone 3 tasks are fully done.**

**Estimated Difficulty:** Medium
**Estimated Time:** 2 hours

## Files to Modify:
```text
/grid_engine/ring_grid.py
```

## Implementation Steps:
1. After the base tier assignment, compute height-variance within each coarse cell's footprint.
2. If variance exceeds a tuned threshold, re-subdivide that one cell into 4 sub-cells at the next-finer tier's resolution (a simple quadtree-style one-level split is enough — you do not need the full production quadtree from the master blueprint).
3. Build one hand-crafted validation scene: a flat road patch at 8m and a pothole-like bump at 15–20m; assert the far-away bump ends up with smaller cells than the closer flat patch — this single test is your entire novelty argument for this task, make sure it's rock-solid before demo day.

## Tests Required:
- `test_far_bump_subdivides_finer_than_close_flat_road`: the core regression test — the direct opposite of what a pure distance rule would produce.

## Verification:
```bash
pytest grid_engine/tests/test_risk_adaptive.py -v
```

## Definition of Done:
```text
[ ] Height-variance-triggered subdivision implemented
[ ] Core regression test (far bump beats close flat road) passing
[ ] Fallback: pure distance-tier mode still selectable/default if this destabilizes anything close to demo time
```

## Common Mistakes:
- Attempting this task before TASK-007–009 are rock solid — a half-working "smarter" grid is a worse demo than a fully-working simple one. This is explicitly last-priority for a reason.

---

### TASK-011: Grid Serialization for API Transport

**Objective:** A simple, fast way to hand a computed grid from the Python backend process to the dashboard as JSON.

**Prerequisites:** TASK-007.

**Estimated Difficulty:** Easy
**Estimated Time:** 1 hour

## Files to Create:
```text
/grid_engine/serialize.py
```

## Implementation Steps:
1. Implement `grid_to_json(grid) -> dict`: a flat list of `{x, y, size, height, class_id}` per cell — deliberately simple (no protobuf/binary codec needed at this scope; that's the full production blueprint's job).
2. Keep the payload small: round floats to 2 decimal places, drop any field the frontend doesn't actually render.
3. Benchmark serialization time on a full frame — if it's a meaningful fraction of your frame budget, that's useful to know before TASK-013.

## Tests Required:
- `test_serialize_roundtrip_cell_count`: JSON output has exactly as many entries as the input grid has cells.

## Verification:
```bash
pytest grid_engine/tests/test_serialize.py -v
```

## Definition of Done:
```text
[ ] Serialization implemented, simple JSON, no over-engineering
[ ] Serialization time benchmarked and small relative to frame budget
```

## Common Mistakes:
- Building a compressed binary protocol at this scope "to be impressive" — it adds real integration risk for a metric nobody's judging you on directly; save that ambition for the documented future-work section instead.

---

## MILESTONE 4: Backend API & Live Playback Loop

**Goal:** A running service that replays the demo sequence through perception → grid engine at a real, measured frame rate, and exposes it over HTTP for the dashboard to consume.

**Estimated time:** ~5 hours (TASK-012 → TASK-014)

### TASK-012: FastAPI Service Scaffolding

**Objective:** Minimal backend exposing frame and metrics endpoints — no database, no auth, no queue.

**Prerequisites:** TASK-006, TASK-011.

**Estimated Difficulty:** Easy
**Estimated Time:** 1 hour

## Install:
```bash
pip install fastapi "uvicorn[standard]"
```

## Files to Create:
```text
/app/backend/main.py
```

## Implementation Steps:
1. Scaffold FastAPI app with `GET /health` returning `{"status": "ok"}`.
2. Add CORS middleware open to `localhost` origins only (dashboard runs locally too — no need for anything stricter at this scope).
3. Confirm it runs and is reachable before building the real endpoints in TASK-013/014.

## Tests Required:
- `test_health_endpoint`.

## Verification:
```bash
uvicorn app.backend.main:app --reload
curl http://localhost:8000/health
```

## Definition of Done:
```text
[ ] FastAPI app runs locally
[ ] Health endpoint works
[ ] CORS allows the dashboard's local origin
```

## Common Mistakes:
- Adding auth/JWT "because the full blueprint has it" — irrelevant at this scope and pure time cost; skip it entirely and say so plainly if asked.

---

### TASK-013: Playback Loop & Pipeline Orchestration

**Objective:** The actual live loop — pull next frame, classify, project, cache latest result — timed against a target frame rate.

**Prerequisites:** TASK-012.

**Estimated Difficulty:** Medium
**Estimated Time:** 2 hours

## Files to Create:
```text
/app/backend/playback.py
```

## API Changes:
```
GET  /frame                 → latest grid, JSON (TASK-011's format)
GET  /frame/uniform         → latest frame's uniform-grid baseline (TASK-009), for the comparison view
POST /playback/start        → begin/resume looping through the demo sequence
POST /playback/stop         → pause
POST /playback/speed?x=1.0  → playback speed multiplier
```

## Implementation Steps:
1. Implement a background loop (thread or asyncio task) that: gets the next frame index (looping back to 0 at the end of the sequence), calls `classify_frame`, calls `RingGrid.project`, serializes, stores as "latest" — repeat, paced to a target interval.
2. Track actual achieved rate vs. target — don't fake a smooth FPS number if the loop is actually running slower; report reality (this feeds TASK-014).
3. Also compute and cache the uniform-grid version of the same frame (TASK-009's baseline) for the comparison view, since recomputing it live for every dashboard poll would be wasteful.
4. Wire the three playback control endpoints to simple in-memory state — no persistence needed.

## Tests Required:
- `test_playback_loops_at_end_of_sequence`: frame index wraps back to 0 rather than crashing.
- `test_frame_endpoint_returns_latest_after_start`.

## Verification:
```bash
uvicorn app.backend.main:app --reload
curl -X POST http://localhost:8000/playback/start
curl http://localhost:8000/frame | head -c 300
pytest app/backend/tests/test_playback.py -v
```

## Definition of Done:
```text
[ ] Background playback loop implemented, loops cleanly at sequence end
[ ] /frame and /frame/uniform both return real, current data
[ ] Playback control endpoints work
[ ] Achieved rate is genuinely measured, not assumed
```

## Common Mistakes:
- Blocking the FastAPI event loop with the playback work so `/frame` requests queue up behind it — run playback in a separate thread/task so the API stays responsive for polling.

---

### TASK-014: Metrics Endpoint & Latency Instrumentation

**Objective:** Real FPS/latency/memory numbers, broken down by pipeline stage — this is your second headline metric and it needs to survive a judge asking "how did you measure that?"

**Prerequisites:** TASK-013, TASK-009.

**Estimated Difficulty:** Medium
**Estimated Time:** 2 hours

## Files to Create:
```text
/app/backend/metrics.py
```

## API Changes:
```
GET /metrics
  Response: {
    "fps": float,
    "latency_ms": {"classify": float, "project": float, "serialize": float, "total": float},
    "memory_saved_pct": float,
    "point_count": int,
    "backend": "groundtruth" | "model"
  }
```

## Implementation Steps:
1. Instrument each stage of TASK-013's loop with `time.perf_counter()` timestamps, keep a rolling window (last ~30 frames) of per-stage timings.
2. Compute `fps` from the rolling window's actual achieved rate, not the configured target.
3. Pull `memory_saved_pct` from TASK-009's comparator, computed on the current frame (cheap enough to do live; if not, cache from the full-sequence run in TASK-009 and label it as "sequence-average" explicitly).
4. Include which perception backend is currently active (`groundtruth` vs `model`) directly in the response — this keeps your dashboard honest about what's actually running, matching TASK-006's logging.

## Logic / Business Rules:
- `latency_ms.total` must be the full pipeline latency (classify + project + serialize), not inference-only — same principle the full production blueprint insists on, and for the same reason: inference-only numbers understate what a judge would actually experience watching the live FPS counter.

## Tests Required:
- `test_metrics_reports_full_pipeline_latency_not_stage_only`: assert `total` ≈ sum of the sub-stages, not just the classify time.
- `test_metrics_backend_field_matches_active_backend`.

## Verification:
```bash
curl http://localhost:8000/metrics
pytest app/backend/tests/test_metrics.py -v
```

## Definition of Done:
```text
[ ] /metrics implemented, rolling-window FPS computed from real timings
[ ] Per-stage latency breakdown present
[ ] memory_saved_pct sourced from real computed data, clearly labeled if it's sequence-average vs. per-frame
[ ] Active backend surfaced in the response
```

## Common Mistakes:
- Reporting only inference time as "latency" — this is the single easiest way to get caught overstating performance if a judge asks a follow-up question about real frame rate.

---

## MILESTONE 5: Real-Time Dashboard

**Goal:** The visual proof — a live, color-coded, genuinely non-uniform grid render, a side-by-side comparison against the uniform baseline, and a metrics panel a judge can watch update in real time.

**Estimated time:** ~9 hours (TASK-015 → TASK-018)

### TASK-015: Frontend Scaffolding & Live Polling

**Objective:** Get a page on screen polling the backend, before investing in the fancier grid renderer.

**Prerequisites:** TASK-012 (health endpoint working).

**Estimated Difficulty:** Easy
**Estimated Time:** 2 hours

## Install:
```bash
# Pick ONE and commit — see Common Mistakes
npm create vite@latest app/frontend -- --template react-ts
# OR, if frontend time is short:
pip install streamlit plotly
```

## Files to Create:
```text
/app/frontend/src/App.tsx           # if React path
/app/frontend/src/lib/apiClient.ts
```

## Implementation Steps:
1. Scaffold the chosen frontend approach.
2. Implement a polling hook/loop hitting `GET /frame`, `GET /frame/uniform`, and `GET /metrics` on a ~200–300ms interval.
3. Render raw JSON on screen first (a plain table or console log is fine) just to prove the live data flow works end-to-end before building the actual visual renderer in TASK-016.

## Tests Required:
- Manual: confirm polling updates visibly as playback runs.

## Verification:
```bash
npm run dev        # React path
# or
streamlit run app/frontend/app.py   # Streamlit path
```

## Definition of Done:
```text
[ ] Frontend scaffolded, one approach chosen and committed to
[ ] Live polling against /frame and /metrics confirmed working
```

## Common Mistakes:
- Debating React-vs-Streamlit for an hour instead of picking one by a simple rule: if someone on the team already knows React/deck.gl well, use it (looks more impressive); otherwise Streamlit/Plotly gets you a working visual faster and more reliably. Decide in 5 minutes and move on.

---

### TASK-016: 🔬 Variable-Resolution Grid Renderer

**Objective:** Render the actual grid with real variable cell sizes — this is the single visual a judge needs to see to understand the novelty.

**Prerequisites:** TASK-015, TASK-011.

**Estimated Difficulty:** Hard
**Estimated Time:** 3 hours

## Install:
```bash
npm install @deck.gl/react @deck.gl/layers @deck.gl/core   # React path
# or for the Streamlit path: reuse plotly's go.Scatter/go.Heatmap with variable marker sizes,
# or a simple matplotlib/PIL rasterization refreshed on each poll
```

## Files to Create:
```text
/app/frontend/src/components/GridRenderer.tsx   # (or grid_renderer.py for Streamlit)
```

## Implementation Steps:
1. For each cell in the polled `/frame` response, draw a rectangle at `(x, y)` sized by `size` (this is the field that must vary — if every rectangle on screen is the same size, something upstream is broken, stop and check TASK-007 before going further).
2. Color by `class_id` using a small fixed palette (reuse the same 4–5 colors everywhere — legend, this renderer, the comparison view).
3. Optional but valuable: extrude/shade by `height` for a pseudo-2.5D look (deck.gl `PolygonLayer` with `getElevation`, or a simple color-intensity proxy for height in the simpler path).
4. Confirm visually, live, against the real demo sequence: cells should be visibly tiny near the origin and visibly large toward the edges.

## Tests Required:
- `test_rendered_cell_sizes_vary`: given a mocked multi-tier frame, assert rendered rectangle sizes are not all identical (regression guard against silently reverting to a uniform look).

## Verification:
```bash
npm test -- GridRenderer   # or manual visual check for the Streamlit path
```

## Definition of Done:
```text
[ ] Grid renders with genuinely variable cell sizes sourced from real backend data
[ ] Color-coding by class matches the shared palette
[ ] Visually confirmed against a live playback run: fine near-field, coarse far-field
```

## Common Mistakes:
- Faking visual variability with a screen-space/camera-distance trick instead of using the real `size` field from the backend — this is the exact anti-pattern the full production blueprint warns against, and it's just as damaging at hackathon scale: it would look identical in a demo while proving nothing about your actual grid engine.

---

### TASK-017: Uniform-vs-Adaptive Comparison View

**Objective:** Put the memory-savings claim next to a picture, not just a number — this is the single highest-leverage visual for judging.

**Prerequisites:** TASK-016, TASK-013 (`/frame/uniform`).

**Estimated Difficulty:** Medium
**Estimated Time:** 2 hours

## Files to Create:
```text
/app/frontend/src/components/ComparisonView.tsx
```

## Implementation Steps:
1. Add a toggle or side-by-side split: left pane renders `/frame/uniform` through the same `GridRenderer`, right pane renders `/frame` (adaptive) — same frame, same colors, different grid.
2. Overlay the `memory_saved_pct` number directly on this view, large and readable — don't bury it in the metrics panel alone.
3. Confirm both panes stay in sync on the same underlying frame (a mismatch here — comparing different frames — would be misleading and is worth explicitly guarding against).

## Tests Required:
- `test_comparison_panes_use_same_frame_timestamp`.

## Verification:
```bash
npm test -- ComparisonView
```

## Definition of Done:
```text
[ ] Side-by-side (or toggle) comparison implemented
[ ] Both panes confirmed synced to the same frame
[ ] Memory-saved percentage displayed prominently on this view
```

## Common Mistakes:
- Showing the uniform grid at a coarser resolution than 5cm "so it renders faster" — this quietly weakens your own comparison; keep the baseline honest at the same finest resolution the problem statement specifies, even if it's slower to render (cache it if needed, per TASK-013).

---

### TASK-018: Metrics Panel & Legend

**Objective:** The always-visible readout of FPS, latency breakdown, and class legend — small, but this is what makes the dashboard read as a real instrument rather than a toy animation.

**Prerequisites:** TASK-017, TASK-014.

**Estimated Difficulty:** Medium
**Estimated Time:** 2 hours

## Files to Create:
```text
/app/frontend/src/components/MetricsPanel.tsx
/app/frontend/src/components/Legend.tsx
```

## Implementation Steps:
1. `MetricsPanel.tsx`: live FPS number, small latency breakdown (classify/project/serialize/total), point count, active-backend indicator — all pulled straight from `/metrics`, refreshed on the same poll interval as the grid.
2. `Legend.tsx`: one swatch per class, using the exact same color mapping as `GridRenderer` (import from one shared constants file — don't define colors twice).
3. Add a simple color-band on the FPS number (e.g., green above your target rate, amber/red below) so the live number reads at a glance during the demo.
4. Basic loading/empty state: if `/metrics` hasn't returned yet, show a neutral placeholder rather than a broken/zeroed panel — small polish, cheap to add, avoids an awkward blank flash right as the demo starts.

## Tests Required:
- `test_legend_and_renderer_share_same_color_source`.

## Verification:
```bash
npm test -- MetricsPanel Legend
```

## Definition of Done:
```text
[ ] Metrics panel implemented, live-updating
[ ] Legend implemented, colors shared with the grid renderer (single source, not duplicated)
[ ] FPS color-banding implemented
[ ] Loading/empty states handled, no broken-looking flash on load
```

## Common Mistakes:
- Defining the color palette inline in the grid renderer during TASK-016 and never factoring it out here — leaves two color sources that will visibly disagree the moment someone tweaks one and forgets the other.

---

## MILESTONE 6: Metrics, Packaging & Demo Readiness

**Goal:** Turn a working pipeline into a winnable demo — a frozen best-case sequence, honestly-reported numbers, a rehearsed script, and slides that map every claim back to the actual problem statement.

**Estimated time:** ~5 hours (TASK-019 → TASK-021)

### TASK-019: Full-Sequence Benchmark Run & Report

**Objective:** One clean, reproducible run producing every number that will appear on a slide.

**Prerequisites:** TASK-018, TASK-014.

**Estimated Difficulty:** Easy
**Estimated Time:** 1.5 hours

## Files to Create:
```text
/docs/reports/FINAL_METRICS_REPORT.md
```

## Implementation Steps:
1. Run the full pipeline over the entire trimmed demo sequence once, uninterrupted, capturing: average FPS, p95 total latency, memory-saved % (average and worst-case across frames), point count per frame, active backend used.
2. If ground-truth labels are available for a held-out check (even a coarse one — e.g., comparing model-backend predictions against ground truth on a subset if TASK-005's model path is live), compute a simple per-class accuracy/confusion count — even an approximate number beats having nothing for the "accuracy across distances" requirement.
3. Write `docs/reports/FINAL_METRICS_REPORT.md` with every number, labeled with exactly how and on what hardware it was measured — this is the document you hand a judge who asks for specifics.

## Tests Required:
- N/A — this is a benchmark/reporting task, not new functionality.

## Verification:
```bash
python app/backend/scripts/run_full_benchmark.py --sequence data/demo_sequence
```

## Definition of Done:
```text
[ ] Full-sequence benchmark run completed uninterrupted
[ ] docs/reports/FINAL_METRICS_REPORT.md written with clearly-labeled, honest numbers
[ ] Any accuracy figures included are labeled with their actual measurement method, not asserted
```

## Common Mistakes:
- Cherry-picking your best 10-second window's FPS as "the number" — report the full-sequence average and be ready to show the raw log if asked.

---

### TASK-020: Demo Script, Fixed Segment & Fallback Recording

**Objective:** Make the live demo short, reliable, and impressive — and have an absolute fallback if live playback breaks on stage.

**Prerequisites:** TASK-019.

**Estimated Difficulty:** Easy
**Estimated Time:** 1.5 hours

## Files to Create:
```text
/docs/DEMO_SCRIPT.md
/docs/demo_recording.mp4   # captured, not scripted-but-unrun
```

## Implementation Steps:
1. Identify the exact ~30–60 second window in your demo sequence with the richest mix of classes and the clearest variable-cell-size moment (ideally the same "hero moment" chosen back in TASK-002).
2. Write `DEMO_SCRIPT.md`: literal steps — start backend, start playback at that frame offset, what to point at on screen and in what order (grid renderer → comparison view → metrics panel), what to say for each.
3. **Actually record a full successful run** of this exact script as a fallback video — do this at least an hour before you think you'll need it, not five minutes before your slot.
4. Rehearse live at least twice against the real running system, timed.

## Tests Required:
- N/A — verification is the rehearsal itself.

## Verification:
- Full dry run against the live system, timed, by someone who didn't write the script.

## Definition of Done:
```text
[ ] Best demo window identified and hardcoded as the default playback start point
[ ] DEMO_SCRIPT.md written, step-by-step
[ ] Fallback video recorded from an actual successful run
[ ] Live-rehearsed at least twice, timed
```

## Common Mistakes:
- Recording the fallback video from memory/description instead of an actual captured run — if it's not a real recording of the real system, it's not a functioning fallback.

---

### TASK-021: Slide Deck & Novelty-to-Requirement Mapping

**Objective:** Close the loop explicitly back to Problem Statement 26053, and show the judges you know exactly what you cut and why.

**Prerequisites:** TASK-020, TASK-019.

**Estimated Difficulty:** Easy
**Estimated Time:** 2 hours

## Files to Create:
```text
/docs/SLIDES.md   (or the actual slide file, whichever tool the team uses)
/docs/NOVELTY_TO_REQUIREMENT_MAPPING.md
```

## Implementation Steps:
1. Build 4–6 slides: Problem → Architecture (the 3-box diagram from `docs/ARCHITECTURE.md`) → Novelty (adaptive grid, real cell-size difference visible on screen) → Live demo → Metrics (the two headline numbers from TASK-019) → What's next (point at the full production roadmap's milestone structure as evidence of understanding the deployment path, without pretending you built it).
2. Write `NOVELTY_TO_REQUIREMENT_MAPPING.md`: one row per problem-statement bullet (terrain analysis, object detection, adaptive spatial representation, alignment/data-loss guarantee, real-time visualization, memory reduction evidence, latency/FPS evidence, accuracy across distances) mapped to the exact task that answers it — mirrors the full blueprint's TASK-059 pattern, scaled down.
3. Explicitly list what was cut (edge deployment, domain adaptation, confidence-tagged cells, swarm codec, security/CI/CD) as a short "future work" slide — stated confidently as scope discipline, not apologized for.

## Tests Required:
- N/A — documentation task.

## Verification:
- Read-through by a teammate who wasn't in the room for a given track's work, confirming every claim on a slide is backed by something actually built and measured.

## Definition of Done:
```text
[ ] Slide deck complete, 4-6 slides, demo-paced
[ ] Novelty-to-requirement mapping covers every bullet in the original problem statement
[ ] Explicit, unapologetic "what we cut and why" slide included
[ ] Every claim on every slide double-checked against something actually built/measured
```

## Common Mistakes:
- Overclaiming in slides beyond what TASK-019's report actually measured — a single caught overclaim costs more credibility with a technical panel than an honestly-scoped MVP ever will.

---

**End of Hackathon Blueprint.** All 6 milestones, TASK-001 through TASK-021, are specified. The two things to protect above everything else if time runs short: TASK-007/008/009 (the grid engine is genuinely non-uniform and provably correct) and TASK-016/017 (that non-uniformity is genuinely visible on screen, next to its own memory-savings number). Everything else — including the model-backend path, the stretch risk-adaptive rule, and dashboard polish — is negotiable against the clock; those two are not.
