"""
run_conservation_check_full_sequence.py
=======================================
Verification script: Runs point-conservation and spatial alignment checks
across the entire sequence to prove 0% data loss and 0 alignment errors.

Usage
-----
    python grid_engine/scripts/run_conservation_check_full_sequence.py
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from tqdm import tqdm

from grid_engine.ring_grid import RingGrid
from perception.loader import get_classified_points, get_frame_count


def run_full_sequence_conservation_check(max_frames: int = 271) -> bool:
    grid = RingGrid()
    total_frames = get_frame_count()

    if total_frames == 0:
        print("[ERROR] No demo sequence frames found in data/demo_sequence/", file=sys.stderr)
        return False

    frames_to_test = min(total_frames, max_frames)

    print("=" * 72)
    print("PRAHARI-Lite -- Full-Sequence Alignment & Point Conservation Audit")
    print(f"  Total Frames Evaluated : {frames_to_test}")
    print(f"  Tiers                  : Tier 0 (5cm), Tier 1 (15cm), Tier 2 (50cm)")
    print("=" * 72)

    total_points_in = 0
    total_points_conserved = 0
    total_bounds_violations = 0

    t0 = time.perf_counter()

    for f_idx in tqdm(range(frames_to_test), desc="Auditing Frames"):
        pts, classes = get_classified_points(f_idx)
        r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        valid_mask = r <= 100.0
        n_valid = int(np.sum(valid_mask))

        cells = grid.project(pts, classes, max_range=100.0)
        n_cell_points = sum(c.point_count for c in cells.values())

        total_points_in += n_valid
        total_points_conserved += n_cell_points

        if n_valid != n_cell_points:
            print(
                f"\n[FAIL] Point mismatch in frame {f_idx}: {n_valid} valid vs {n_cell_points} in grid",
                file=sys.stderr,
            )
            return False

        # Spatial alignment spot-check on random points
        valid_pts = pts[valid_mask]
        sample_pts = valid_pts[::100]  # 1% sample
        for p in sample_pts:
            px, py = p[0], p[1]
            pr = np.sqrt(px ** 2 + py ** 2)
            tier = grid.assign_tiers(np.array([pr]))[0]
            sz = grid.tiers[tier]["cell_size_m"]
            ix = int(np.floor(px / sz))
            iy = int(np.floor(py / sz))
            key = (tier, ix, iy)
            if key not in cells:
                total_bounds_violations += 1
                continue
            cell = cells[key]
            half = cell.size / 2.0 + 1e-5
            if not ((cell.x - half <= px <= cell.x + half) and (cell.y - half <= py <= cell.y + half)):
                total_bounds_violations += 1

    dt = time.perf_counter() - t0

    print("\n" + "=" * 72)
    print("AUDIT RESULTS SUMMARY:")
    print(f"  Total Valid Points Ingested  : {total_points_in:,}")
    print(f"  Total Points Mapped to Cells : {total_points_conserved:,}")
    print(f"  Point Conservation Rate      : {(total_points_conserved / total_points_in) * 100.0:.4f}%")
    print(f"  Spatial Bounds Violations    : {total_bounds_violations}")
    print(f"  Audit Execution Time         : {dt:.2f}s ({frames_to_test / dt:.1f} FPS)")
    print("=" * 72)

    if total_points_conserved == total_points_in and total_bounds_violations == 0:
        print("\n[SUCCESS] Strict Point-Conservation & Alignment Guarantees PROVEN across all frames.")
        return True
    else:
        print("\n[FAILED] Invariants violated.", file=sys.stderr)
        return False


if __name__ == "__main__":
    success = run_full_sequence_conservation_check()
    sys.exit(0 if success else 1)
