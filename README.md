# PRAHARI

A deployable prototype for DRDO problem statement **26053 — Adaptive Variable Resolution
2.5D LiDAR Mapping for Dynamic Environment Perception**.

One pipeline, four stages, each timed every frame:

```
perception  →  variable-resolution grid  →  [ optional transmit tier ]  →  codec  →  browser
```

The dashboard renders exactly what the codec produced — not a separate, roomier debug
view — so a codec bug shows up as a broken map, not as an optimistic number in a caption.

## Quick start

```bash
./run.sh
```

Open **http://localhost:8000**. That's the whole setup: a virtualenv, `pip install`, and
`uvicorn`. No GPU, no dataset, no checkpoint required — the synthetic sensor model runs the
full pipeline standalone.

Windows / no bash: `python -m venv .venv && .venv\Scripts\activate && pip install -r
requirements.txt && uvicorn backend.main:app --port 8000`.

## What's actually being demonstrated

| PS requirement | Where |
|---|---|
| Terrain vs static obstacle vs moving object segmentation | `backend/scene.py` (synthetic) or your FRNet checkpoint via `backend/perception.py` |
| Walls, poles, pedestrians, vehicles named explicitly | `PS_CATEGORIES` in `backend/config.py`, shown per-category in the Evidence tab |
| 5 cm cells inside 10 m, coarsening toward 50 cm at 100 m | `backend/grid_engine.py` — the level table is published at `/api/system` and shown live in the status bar, not just asserted |
| 3D → 2.5D projection without alignment errors or data loss | Points are quantised once onto the finest grid and Morton-sorted; every quadtree node is a contiguous run of that sorted array, so a point cannot fall between two cells of different resolution. Tested directly in `test_grid_engine.py::test_every_point_lies_inside_the_cell_that_claims_it` |
| Real-time dashboard, color-coded, showing memory reduction | The Live tab — semantic / elevation / confidence / cell-size shading, memory bars against a dense 2.5D grid, dense 3D voxels, and the raw cloud |
| Low latency / high FPS | Live throughput panel — 10 Hz target, rolling FPS, per-stage timing, a frame-budget line on the sparkline |
| Accuracy across varying distances | Evidence tab, **read from your training notebook's evidence export** — never computed by this server |

## The one thing to say honestly when you present this

**The perception source is synthetic unless a checkpoint is attached**, and the dashboard
never lets that be ambiguous: every frame carries a source badge, red for synthetic, green
for a live checkpoint, and the badge is driven by the same field the pipeline uses
internally — there's no separate "demo mode" flag to forget to flip back.

**Full-resolution transmission does not fit a constrained tactical link.** At full detail
this map runs about 6 Mbit/s at 10 Hz. The transmit-tier control on the Live tab is the
answer: it coarsens everything for the remote link *except* cells whose resolution was set
by a risk rule (hazard, degraded sensing), so a compressed, tiled road still arrives with
its potholes at full resolution. The 40 cm tier lands near 1.2 Mbit/s; 80 cm gets under
1 Mbit/s.

**Every accuracy number on the Evidence tab comes from the notebook, or it says
"Not measured."** This server computes latency, throughput, cell counts and memory
directly — those are real measurements of real code, made here. It does not compute
mIoU, AP, or any classification accuracy, because it has no ground truth to check against.
That number can only come from the training pipeline, and the Evidence tab reads it from
there rather than inventing a plausible one.

## Attaching your FRNet checkpoint

By default `backend/perception.py` runs `SyntheticPerception`, an analytic ray-caster
scene (walls, poles, pedestrians, vehicles, hazards) built for demonstrating the full
pipeline without a dataset. To run your real segmentation model instead:

```bash
export PRAHARI_CKPT=/path/to/frnet_indian_finetune/latest.pth
export PRAHARI_CFG=/path/to/frnet-semantickitti_seg.py
export PRAHARI_SCANS=/path/to/a/directory/of/*.bin
./run.sh
```

All three must be set together — a partially configured checkpoint falls back to the
synthetic model and says exactly which variable is missing, rather than silently running
synthetic data while the badge implies otherwise. `mmdet3d` and `torch` are only imported
on this path, so the synthetic path has no such dependency.

The confidence signal that drives risk-adaptive subdivision is pulled from the same
frustum/auxiliary head the training notebook's Phase 5 exposes — see
`CheckpointPerception._frustum_confidence` in `backend/perception.py`. If your FRNet
revision names that head differently, that's the one place to edit.

## Connecting the Evidence tab to your training run

Run the TRINETRA training notebook through Phase 7. It writes
`trinetra_evidence_export.json` plus `ps_category_table_*.json` and
`distance_stratified_*.json` alongside it. Point the server at that directory:

```bash
export PRAHARI_EVIDENCE=/path/to/evidence/trinetra_evidence_export.json
```

If unset, the server looks for `./evidence/trinetra_evidence_export.json` next to
`backend/`, which is empty in this delivery — the Evidence tab will show every block as
"Not measured" until you point it at a real export, which is the correct behaviour, not
a bug.

## Project layout

```
backend/
  config.py       geometry, classes, hazard set, resolution schedule — single source of truth
  scene.py        synthetic 64-beam sensor model + analytic scene
  grid_engine.py  the variable-resolution quadtree (the core) + the transmit-tier decimator
  codec.py        PRH1 wire format: what is sent is what is drawn
  perception.py   synthetic sensor model / real FRNet checkpoint, behind one interface
  metrics.py      reads the training notebook's evidence export; never invents a number
  main.py         FastAPI app, WebSocket stream, REST endpoints
  tests/          52 tests: alignment, foveation, risk-adaptation, codec, transmit tier, API
frontend/
  index.html, css/app.css
  js/wire.js        binary frame decoder (mirrors codec.py exactly)
  js/palette.js     class colours, elevation/confidence/resolution ramps
  js/viewport.js    the map canvas: batched cell fills, range rings, hazard marks, detections
  js/telemetry.js   throughput, stage timing, memory bars, link budget, risk breakdown
  js/evidence.js    renders measured/not-measured blocks with provenance
  js/app.js         bootstrap, WebSocket wiring, controls
tools/
  verify_wire.mjs   runs the browser's own decoder against a live server and checks the map
run.sh, requirements.txt
```

## Verifying it yourself

```bash
pip install -r requirements.txt
python -m pytest backend/tests -q          # 52 tests

# with the server running (./run.sh in another terminal):
node tools/verify_wire.mjs                 # decodes real frames with the browser's own code
```

`verify_wire.mjs` matters because it is not a Python test asserting against a Python
codec — it imports the exact `wire.js` the browser loads and decodes real bytes from a
running server, so it catches a mismatch between what the backend writes and what the
frontend reads that a same-language test suite structurally cannot.

## Known limits of this prototype

- The synthetic scene is a fixed rural-road layout with looping traffic. It is built to
  exercise every PS category and both hazard-detection paths (a hazard beyond the fovea,
  a dust-degraded confidence sector) — it is not a dataset and proves nothing about
  real-world accuracy.
- `CheckpointPerception` expects an MMDetection3D-style FRNet model with a named
  frustum/auxiliary head; a different checkpoint layout will need a small adapter change
  in `backend/perception.py`.
- The detection boxes shown alongside segmentation are read from the synthetic scene's
  ground truth when no checkpoint is attached; wiring a real PillarNet checkpoint into
  `perception.py` is not yet implemented (the segmentation path is).
- Benchmarks above are this machine's CPU, not the Jetson Orin Nano the deck targets —
  the Evidence tab keeps those two apart deliberately, and on-device numbers only appear
  there once your notebook's Phase 6 has produced them.
