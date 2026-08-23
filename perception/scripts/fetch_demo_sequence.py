"""
fetch_demo_sequence.py
======================
Acquires and trims the real demo LiDAR sequence used throughout PRAHARI-Lite.

Data sources
------------
* SemanticKITTI labels (179 MB, no login required):
      http://semantic-kitti.org/assets/data_odometry_labels.zip
* KITTI Velodyne LiDAR (Public AWS S3 bucket - 2011_09_30_drive_0016_sync):
      https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0016/2011_09_30_drive_0016_sync.zip

Sequence 04 corresponds to KITTI Raw: 2011_09_30 Drive 0016 (271 frames).

Usage
-----
    # Download and prepare original real data:
    python perception/scripts/fetch_demo_sequence.py

    # Download only labels:
    python perception/scripts/fetch_demo_sequence.py --labels-only

    # Download only velodyne:
    python perception/scripts/fetch_demo_sequence.py --velodyne-only

    # Generate synthetic placeholder data for offline dev:
    python perception/scripts/fetch_demo_sequence.py --synthetic
"""

import argparse
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration -- matches the blueprint spec
# ---------------------------------------------------------------------------
TARGET_SEQUENCE: str = "04"
FRAME_RANGE: tuple = (0, 271)            # seq-04 has 271 frames (0-270)
POINTS_PER_FRAME: int = 120_000          # synthetic fallback: pts/frame

LABELS_URL: str = "http://semantic-kitti.org/assets/data_odometry_labels.zip"
LABELS_FILENAME: str = "data_odometry_labels.zip"

VELODYNE_URL: str = (
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
    "2011_09_30_drive_0016/2011_09_30_drive_0016_sync.zip"
)
VELODYNE_FILENAME: str = "2011_09_30_drive_0016_sync.zip"

# Paths (relative to project root -- run this script from PRAHARI/)
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR      = ROOT_DIR / "data" / "raw"
DEMO_DIR     = ROOT_DIR / "data" / "demo_sequence"
VELODYNE_SRC = RAW_DIR / "sequences" / TARGET_SEQUENCE / "velodyne"
LABELS_SRC   = RAW_DIR / "sequences" / TARGET_SEQUENCE / "labels"
VELODYNE_DST = DEMO_DIR / "velodyne"
LABELS_DST   = DEMO_DIR / "labels"


# ---------------------------------------------------------------------------
# Resumable, robust HTTP downloader
# ---------------------------------------------------------------------------

def download_resumable(
    url: str,
    dest: Path,
    max_retries: int = 20,
    timeout: int = 30,
) -> bool:
    """
    Download url to dest with automatic HTTP Range resume and exponential backoff.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        existing_size = temp_dest.stat().st_size if temp_dest.exists() else 0
        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        try:
            print(f"[{attempt}/{max_retries}] Connecting to {url} (resuming from {existing_size / (1024*1024):.2f} MB) ...")
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code not in (200, 206, 416):
                    r.raise_for_status()

                if r.status_code == 416:
                    # Requested range not satisfiable -> file is already fully downloaded
                    print(f"File already complete: {temp_dest.name}")
                    break

                total_size = int(r.headers.get("content-length", 0)) + existing_size
                mode = "ab" if existing_size > 0 and r.status_code == 206 else "wb"
                if mode == "wb":
                    existing_size = 0

                with open(temp_dest, mode) as f, tqdm(
                    total=total_size,
                    initial=existing_size,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            # Download finished successfully
            break

        except (requests.RequestException, IOError) as exc:
            print(f"\n[WARN] Connection issue: {exc}. Retrying in {min(2 ** attempt, 15)}s...", file=sys.stderr)
            time.sleep(min(2 ** attempt, 15))
    else:
        print(f"[ERROR] Failed to download {url} after {max_retries} attempts.", file=sys.stderr)
        return False

    if temp_dest.exists():
        if dest.exists():
            dest.unlink()
        temp_dest.rename(dest)
        print(f"[OK] Downloaded: {dest} ({dest.stat().st_size / (1024*1024):.2f} MB)")
        return True

    return False


# ---------------------------------------------------------------------------
# Extraction & Conversion Helpers
# ---------------------------------------------------------------------------

def extract_labels_from_zip(zip_path: Path, seq: str, src_dir: Path) -> bool:
    """Extract SemanticKITTI labels for seq from zip into src_dir."""
    prefix = f"dataset/sequences/{seq}/labels/"
    src_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {prefix} from {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith(prefix) and m.endswith(".label")]
        if not members:
            print(f"[ERROR] No labels found in {zip_path} matching {prefix}", file=sys.stderr)
            return False
        for member in tqdm(members, desc="Extracting labels"):
            data = zf.read(member)
            out_file = src_dir / Path(member).name
            out_file.write_bytes(data)
    print(f"  Extracted {len(members)} label files -> {src_dir}")
    return True


def extract_velodyne_from_sync_zip(zip_path: Path, src_dir: Path) -> bool:
    """Extract velodyne point clouds from KITTI raw sync.zip and standardize filenames."""
    src_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting Velodyne scans from {zip_path.name} ...")
    
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m for m in zf.namelist() 
            if "velodyne_points/data/" in m and m.endswith(".bin")
        ]
        members.sort()
        if not members:
            print(f"[ERROR] No velodyne .bin files found in {zip_path}", file=sys.stderr)
            return False

        print(f"  Found {len(members)} point cloud scans in archive.")
        for idx, member in enumerate(tqdm(members, desc="Extracting velodyne")):
            data = zf.read(member)
            # Standardize filename to 6 digits (000000.bin, 000001.bin, ...)
            out_file = src_dir / f"{idx:06d}.bin"
            out_file.write_bytes(data)

    print(f"  Standardized {len(members)} velodyne scans -> {src_dir}")
    return True


def trim_and_copy(src: Path, dst: Path, frame_range: tuple, ext: str) -> list:
    """Copy frames [frame_range[0], frame_range[1]) from src to dst."""
    dst.mkdir(parents=True, exist_ok=True)
    start, end = frame_range
    copied = []
    for i in range(start, end):
        fname = f"{i:06d}.{ext}"
        src_file = src / fname
        dst_file = dst / fname
        if not src_file.exists():
            print(f"  [WARN] Missing frame: {src_file} -- skipping", file=sys.stderr)
            continue
        dst_file.write_bytes(src_file.read_bytes())
        copied.append(fname)
    return copied


def verify_counts(velodyne_dir: Path, labels_dir: Path) -> bool:
    """Return True if frame counts match and are non-zero."""
    bins   = sorted(velodyne_dir.glob("*.bin"))
    labels = sorted(labels_dir.glob("*.label"))
    ok = len(bins) == len(labels) and len(bins) > 0
    if ok:
        print(f"[OK] Frame/label count match: {len(bins)} frames each.")
    else:
        print(
            f"[FAIL] Mismatch: {len(bins)} .bin frames vs {len(labels)} .label files.",
            file=sys.stderr,
        )
    return ok


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def _write_synthetic_bin(path: Path, n_points: int = POINTS_PER_FRAME) -> None:
    rng = np.random.default_rng(seed=int(path.stem))
    pts = rng.standard_normal((n_points, 4)).astype(np.float32)
    pts[:, :2] *= 30.0
    pts[:, 2]   = rng.uniform(-1.0, 2.5, n_points).astype(np.float32)
    pts[:, 3]   = rng.uniform(0.0, 1.0, n_points).astype(np.float32)
    path.write_bytes(pts.tobytes())


def _write_synthetic_label(path: Path, n_points: int = POINTS_PER_FRAME) -> None:
    rng = np.random.default_rng(seed=int(path.stem) + 9999)
    raw_classes = rng.choice(
        [40, 50, 71, 30, 10, 0],
        size=n_points,
        p=[0.40, 0.15, 0.05, 0.05, 0.10, 0.25],
    ).astype(np.uint32)
    path.write_bytes(raw_classes.tobytes())


def generate_synthetic_sequence(
    velodyne_dir: Path,
    labels_dir: Path,
    frame_range: tuple,
    n_points: int = POINTS_PER_FRAME,
) -> None:
    velodyne_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    start, end = frame_range
    print(f"Generating {end - start} synthetic frames ({n_points} pts/frame) ...")
    for i in tqdm(range(start, end), desc="Synthetic frames"):
        _write_synthetic_bin(velodyne_dir / f"{i:06d}.bin", n_points)
        _write_synthetic_label(labels_dir / f"{i:06d}.label", n_points)
    print(f"[OK] Synthetic sequence written to {velodyne_dir.parent}")


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def acquire_original_data(labels_only: bool = False, velodyne_only: bool = False) -> bool:
    """Download and extract real SemanticKITTI labels and real KITTI Velodyne scans."""
    labels_zip = RAW_DIR / LABELS_FILENAME
    velodyne_zip = RAW_DIR / VELODYNE_FILENAME

    # 1. Download & Extract Labels
    if not velodyne_only:
        print("\n--- [1/2] Acquiring SemanticKITTI Labels (179 MB) ---")
        if not labels_zip.exists() or labels_zip.stat().st_size < 170 * 1024 * 1024:
            part_file = RAW_DIR / "data_odometry_labels.zip.part"
            if not part_file.exists() and (RAW_DIR / "data_odometry_labels.zip").exists():
                (RAW_DIR / "data_odometry_labels.zip").rename(part_file)

            ok = download_resumable(LABELS_URL, labels_zip)
            if not ok:
                return False
        else:
            print(f"Labels zip already present: {labels_zip}")

        print("\n--- Extracting Sequence 04 Labels ---")
        if not LABELS_SRC.exists() or len(list(LABELS_SRC.glob("*.label"))) < 271:
            ok = extract_labels_from_zip(labels_zip, TARGET_SEQUENCE, LABELS_SRC)
            if not ok:
                return False
        else:
            print(f"Sequence 04 labels already extracted in {LABELS_SRC}")

    if labels_only:
        return True

    # 2. Download & Extract Velodyne Scans
    print("\n--- [2/2] Acquiring KITTI Velodyne LiDAR (AWS S3, ~1.1 GB) ---")
    if not velodyne_zip.exists() or velodyne_zip.stat().st_size < 1_000_000_000:
        ok = download_resumable(VELODYNE_URL, velodyne_zip)
        if not ok:
            return False
    else:
        print(f"Velodyne zip already present: {velodyne_zip}")

    print("\n--- Extracting & Standardizing Velodyne Scans ---")
    if not VELODYNE_SRC.exists() or len(list(VELODYNE_SRC.glob("*.bin"))) < 271:
        ok = extract_velodyne_from_sync_zip(velodyne_zip, VELODYNE_SRC)
        if not ok:
            return False
    else:
        print(f"Velodyne scans already extracted in {VELODYNE_SRC}")

    return True


def trim_sequence() -> bool:
    """Copy the trimmed frame window from raw -> demo_sequence."""
    print(f"\nPopulating data/demo_sequence with frames {FRAME_RANGE[0]}-{FRAME_RANGE[1]-1} ...")
    bins   = trim_and_copy(VELODYNE_SRC, VELODYNE_DST, FRAME_RANGE, "bin")
    labels = trim_and_copy(LABELS_SRC,   LABELS_DST,   FRAME_RANGE, "label")

    if not bins or not labels:
        print("[ERROR] Failed copying frames to demo_sequence", file=sys.stderr)
        return False

    print(f"  Copied {len(bins)} .bin frames -> {VELODYNE_DST}")
    print(f"  Copied {len(labels)} .label files -> {LABELS_DST}")
    return True


def main(args: argparse.Namespace) -> int:
    print("=" * 60)
    print("PRAHARI-Lite -- fetch_demo_sequence.py")
    print(f"  Sequence : {TARGET_SEQUENCE}")
    print(f"  Frames   : {FRAME_RANGE[0]} - {FRAME_RANGE[1] - 1}")
    print(f"  Mode     : {'synthetic' if args.synthetic else 'real'}")
    print("=" * 60)

    if args.synthetic:
        generate_synthetic_sequence(VELODYNE_DST, LABELS_DST, FRAME_RANGE)
        ok = verify_counts(VELODYNE_DST, LABELS_DST)
        return 0 if ok else 1

    ok = acquire_original_data(labels_only=args.labels_only, velodyne_only=args.velodyne_only)
    if not ok:
        print("\n[ERROR] Failed to acquire original data.", file=sys.stderr)
        return 1

    if args.labels_only or args.velodyne_only:
        print("\n[INFO] Partial acquisition complete.")
        return 0

    if not trim_sequence():
        return 1

    ok = verify_counts(VELODYNE_DST, LABELS_DST)
    if not ok:
        return 1

    print("\n[SUCCESS] Original real LiDAR sequence 04 is fully downloaded and verified!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRAHARI-Lite demo sequence setup")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic placeholder data instead of downloading real data.",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Only download/extract the SemanticKITTI labels; skip velodyne.",
    )
    parser.add_argument(
        "--velodyne-only",
        action="store_true",
        help="Skip label download; just download velodyne.",
    )
    sys.exit(main(parser.parse_args()))
