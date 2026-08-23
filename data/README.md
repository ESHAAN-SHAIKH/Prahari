# data/ — Demo LiDAR Sequence

This directory holds the trimmed SemanticKITTI demo sequence used for
development and the final benchmark run.

> **Not committed to git.** All `*.bin` and `*.label` files are excluded
> by `.gitignore` to keep the repository lightweight.

## Expected layout (after running TASK-002)

```
data/
└── demo_sequence/
    ├── velodyne/   # *.bin  — raw Velodyne point-cloud frames
    └── labels/     # *.label — per-point SemanticKITTI class labels
```

## Download instructions

Run the acquisition script from the project root:

```bash
python perception/scripts/fetch_demo_sequence.py
```

The script downloads and trims the chosen sequence automatically.
See the script's header comments for the exact source URL, sequence ID,
and frame-range rationale.

## Selection criteria

- Sequence: SemanticKITTI sequence **04** (or whichever was chosen — update this
  README when the final selection is made in TASK-002).
- Frame range: approximately 0–400 frames.
- Selection rationale: clear mix of drivable road, static poles/buildings, at
  least one visible pedestrian pass, and at least one vehicle pass — making the
  \"hero moment\" obvious during the live demo.
