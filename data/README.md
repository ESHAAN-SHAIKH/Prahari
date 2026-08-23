# data/ — Demo LiDAR Sequence

This directory holds the trimmed SemanticKITTI demo sequence used for
development and the final benchmark run.

> **Not committed to git.** All `*.bin` and `*.label` files are excluded
> by `.gitignore` to keep the repository lightweight.

---

## Sequence selection (TASK-002)

| Field | Value |
|-------|-------|
| **Dataset** | SemanticKITTI (KITTI Odometry + semantic labels) |
| **Sequence** | `04` |
| **Total frames** | 271 (000000 – 000270, all frames used) |
| **Point density** | ~120 k points / frame (Velodyne HDL-64E) |
| **Selection rationale** | Clear road segment with drivable surface, static buildings and poles, at least one pedestrian pass and one vehicle pass — provides the "hero moment" needed for the live demo's classification showcase |

---

## Data sources

### SemanticKITTI labels (179 MB — auto-downloaded, no account required)
```
http://semantic-kitti.org/assets/data_odometry_labels.zip
```
The acquisition script downloads this automatically.

### KITTI Odometry velodyne point clouds (80 GB — manual download required)
```
https://www.cvlibs.net/datasets/kitti/eval_odometry.php
```
1. Register for a free KITTI account and log in.
2. Download **"Velodyne point clouds (80 GB)"** (`data_odometry_velodyne.zip`).
3. Extract **only** `sequences/04/velodyne/` into `data/raw/sequences/04/velodyne/`.
4. Re-run the acquisition script — it will skip the label re-download.

---

## Quick-start options

```bash
# Option A: Download labels, then trim (velodyne must be placed manually first)
python perception/scripts/fetch_demo_sequence.py

# Option B: Download labels only (velodyne to follow later)
python perception/scripts/fetch_demo_sequence.py --labels-only

# Option C: Synthetic placeholder data — no downloads, immediate dev/CI use
python perception/scripts/fetch_demo_sequence.py --synthetic
```

---

## Expected layout after acquisition

```
data/
├── raw/
│   └── sequences/
│       └── 04/
│           ├── velodyne/    # *.bin — full raw download (not committed)
│           └── labels/      # *.label — extracted from labels.zip (not committed)
│
└── demo_sequence/           # trimmed working copy used by the pipeline
    ├── velodyne/            # 271 × *.bin
    └── labels/              # 271 × *.label
```

---

## Verification

After running the acquisition script:

```bash
pytest perception/tests/test_demo_sequence.py -v
```

All tests must pass before moving to TASK-003.

