"""
fetch_demo_sequence.py
======================
Acquires and trims the demo LiDAR sequence used throughout PRAHARI-Lite.

Data sources
------------
* SemanticKITTI labels (179 MB, no login required):
      http://semantic-kitti.org/assets/data_odometry_labels.zip
* KITTI Odometry velodyne (80 GB, requires free KITTI account):
      https://www.cvlibs.net/datasets/kitti/eval_odometry.php
      -> download "Velodyne point clouds (80 GB)"
      Extract sequences/04/velodyne/ into data/raw/sequences/04/velodyne/

[ASSUMPTION] Using SemanticKITTI sequence 04, all 271 available frames
(000000-000270).  The blueprint targets frames 0-400 but sequence 04
only has 271 -- so we use all of them.  Sequence 04 contains a clear
road segment with cars, pedestrians, and static infrastructure, making
it ideal as the demo "hero" sequence.

Usage
-----
    # Full acquisition (velodyne must already be downloaded manually):
    python perception/scripts/fetch_demo_sequence.py

    # Download only the labels (no velodyne needed yet):
    python perception/scripts/fetch_demo_sequence.py --labels-only

    # Generate synthetic placeholder data for immediate dev/CI:
    python perception/scripts/fetch_demo_sequence.py --synthetic

    # Skip download if files already exist:
    python perception/scripts/fetch_demo_sequence.py --skip-download
"""

import argparse
import sys
import zipfile
from pathlib import Path

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

# Paths (relative to project root -- run this script from PRAHARI/)
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR      = ROOT_DIR / "data" / "raw"
DEMO_DIR     = ROOT_DIR / "data" / "demo_sequence"
VELODYNE_SRC = RAW_DIR / "sequences" / TARGET_SEQUENCE / "velodyne"
LABELS_SRC   = RAW_DIR / "sequences" / TARGET_SEQUENCE / "labels"
VELODYNE_DST = DEMO_DIR / "velodyne"
LABELS_DST   = DEMO_DIR / "labels"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest: Path) -> None:
    """Stream-download url to dest, showing a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading {} -> {} ...".format(url, dest))
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def _extract_sequence_from_zip(zip_path: Path, seq: str, src_dir: Path) -> None:
    """Extract only the labels for seq from the zip into src_dir."""
    prefix = "dataset/sequences/{}/labels/".format(seq)
    src_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting {} from {} ...".format(prefix, zip_path.name))
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith(prefix) and m.endswith(".label")]
        for member in tqdm(members, desc="Extracting labels"):
            data = zf.read(member)
            out_file = src_dir / Path(member).name
            out_file.write_bytes(data)
    print("  Extracted {} label files -> {}".format(len(members), src_dir))


def _trim_and_copy(src: Path, dst: Path, frame_range: tuple, ext: str) -> list:
    """Copy frames [frame_range[0], frame_range[1]) from src to dst."""
    dst.mkdir(parents=True, exist_ok=True)
    start, end = frame_range
    copied = []
    for i in range(start, end):
        fname = "{:06d}.{}".format(i, ext)
        src_file = src / fname
        dst_file = dst / fname
        if not src_file.exists():
            print("  [WARN] Missing frame: {} -- skipping".format(src_file), file=sys.stderr)
            continue
        dst_file.write_bytes(src_file.read_bytes())
        copied.append(fname)
    return copied


def _verify_counts(velodyne_dir: Path, labels_dir: Path) -> bool:
    """Return True if frame counts match and are non-zero."""
    bins   = sorted(velodyne_dir.glob("*.bin"))
    labels = sorted(labels_dir.glob("*.label"))
    ok = len(bins) == len(labels) and len(bins) > 0
    if ok:
        print("[OK] Frame/label count match: {} frames each.".format(len(bins)))
    else:
        print(
            "[FAIL] Mismatch: {} .bin frames vs {} .label files.".format(len(bins), len(labels)),
            file=sys.stderr,
        )
    return ok


# ---------------------------------------------------------------------------
# Synthetic data generation (development fallback)
# ---------------------------------------------------------------------------

def _write_synthetic_bin(path: Path, n_points: int = POINTS_PER_FRAME) -> None:
    """Write a synthetic KITTI .bin point-cloud (x,y,z,intensity) float32."""
    rng = np.random.default_rng(seed=int(path.stem))
    pts = rng.standard_normal((n_points, 4)).astype(np.float32)
    # Realistic-ish spatial spread: x,y +-30 m, z -1 to +2.5 m, intensity 0-1
    pts[:, :2] *= 30.0
    pts[:, 2]   = rng.uniform(-1.0, 2.5, n_points).astype(np.float32)
    pts[:, 3]   = rng.uniform(0.0, 1.0, n_points).astype(np.float32)
    path.write_bytes(pts.tobytes())


def _write_synthetic_label(path: Path, n_points: int = POINTS_PER_FRAME) -> None:
    """Write a synthetic .label file (uint32 per point, SemanticKITTI format).

    Label encoding:  lower 16 bits = semantic id, upper 16 bits = instance id.
    Class distribution:
        40 (road)       ~40%
        50 (building)   ~15%
        71 (pole)        ~5%
        30 (person)      ~5%
        10 (car)        ~10%
         0 (unlabeled)  ~25%
    """
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
    """Generate synthetic .bin and .label files for CI / offline development."""
    velodyne_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    start, end = frame_range
    print("Generating {} synthetic frames ({} pts/frame) ...".format(end - start, n_points))
    for i in tqdm(range(start, end), desc="Synthetic frames"):
        _write_synthetic_bin(velodyne_dir / "{:06d}.bin".format(i), n_points)
        _write_synthetic_label(labels_dir / "{:06d}.label".format(i), n_points)
    print("[OK] Synthetic sequence written to {}".format(velodyne_dir.parent))


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def download_labels(skip_if_exists: bool = False) -> bool:
    """Download and extract SemanticKITTI labels for TARGET_SEQUENCE."""
    zip_dest = RAW_DIR / LABELS_FILENAME

    if skip_if_exists and LABELS_SRC.exists() and any(LABELS_SRC.glob("*.label")):
        print("Labels already present at {} -- skipping download.".format(LABELS_SRC))
        return True

    # Download labels zip
    if not zip_dest.exists():
        try:
            _download_with_progress(LABELS_URL, zip_dest)
        except Exception as exc:
            print("[ERROR] Label download failed: {}".format(exc), file=sys.stderr)
            return False
    else:
        print("Labels zip already cached at {}.".format(zip_dest))

    # Extract
    _extract_sequence_from_zip(zip_dest, TARGET_SEQUENCE, LABELS_SRC)
    return True


def check_velodyne_available() -> bool:
    """Check whether velodyne data for TARGET_SEQUENCE exists locally."""
    bins = list(VELODYNE_SRC.glob("*.bin")) if VELODYNE_SRC.exists() else []
    if bins:
        print("Velodyne data found: {} .bin files at {}".format(len(bins), VELODYNE_SRC))
        return True

    msg = (
        "\n" + "=" * 72 + "\n"
        "MANUAL STEP REQUIRED -- Velodyne point clouds\n"
        "=" * 72 + "\n"
        "The KITTI velodyne archive requires a free account at:\n"
        "  https://www.cvlibs.net/datasets/kitti/eval_odometry.php\n\n"
        "1. Register / log in.\n"
        "2. Download 'Velodyne point clouds (80 GB)' -> data_odometry_velodyne.zip\n"
        "3. Extract ONLY sequences/04/velodyne/ from the zip to:\n"
        "   {}\n\n"
        "Then re-run this script (it will skip the labels re-download).\n"
        "Or run with --synthetic to generate placeholder data now.\n"
        + "=" * 72 + "\n"
    ).format(VELODYNE_SRC)
    print(msg, file=sys.stderr)
    return False


def trim_sequence() -> bool:
    """Copy the trimmed frame window from raw -> demo_sequence."""
    print("\nTrimming frames {}-{} ...".format(FRAME_RANGE[0], FRAME_RANGE[1] - 1))

    bins   = _trim_and_copy(VELODYNE_SRC, VELODYNE_DST, FRAME_RANGE, "bin")
    labels = _trim_and_copy(LABELS_SRC,   LABELS_DST,   FRAME_RANGE, "label")

    if not bins:
        print("[ERROR] No .bin frames were copied.", file=sys.stderr)
        return False
    if not labels:
        print("[ERROR] No .label files were copied.", file=sys.stderr)
        return False

    print("  Copied {} .bin  frames -> {}".format(len(bins), VELODYNE_DST))
    print("  Copied {} .label files -> {}".format(len(labels), LABELS_DST))
    return True


def main(args: argparse.Namespace) -> int:
    print("=" * 60)
    print("PRAHARI-Lite -- fetch_demo_sequence.py")
    print("  Sequence : {}".format(TARGET_SEQUENCE))
    print("  Frames   : {} - {}".format(FRAME_RANGE[0], FRAME_RANGE[1] - 1))
    print("  Mode     : {}".format("synthetic" if args.synthetic else "real"))
    print("=" * 60 + "\n")

    # -- Synthetic path -------------------------------------------------------
    if args.synthetic:
        generate_synthetic_sequence(VELODYNE_DST, LABELS_DST, FRAME_RANGE)
        ok = _verify_counts(VELODYNE_DST, LABELS_DST)
        return 0 if ok else 1

    # -- Real data path -------------------------------------------------------
    if not args.skip_download and not args.velodyne_only:
        ok = download_labels(skip_if_exists=True)
        if not ok:
            return 1

    if args.labels_only:
        print("--labels-only: skipping velodyne check and trim.")
        return 0

    if not check_velodyne_available():
        return 1

    if not trim_sequence():
        return 1

    ok = _verify_counts(VELODYNE_DST, LABELS_DST)
    if not ok:
        return 1

    # Check the "hero moment" heuristic -- we want at least some person (30)
    # and car (10) labels in the first 10 frames.
    print("\nChecking for hero-moment classes (pedestrian + vehicle) ...")
    found_person = False
    found_car    = False
    for i in range(min(10, FRAME_RANGE[1] - FRAME_RANGE[0])):
        lbl_path = LABELS_DST / "{:06d}.label".format(i)
        if lbl_path.exists():
            raw = np.frombuffer(lbl_path.read_bytes(), dtype=np.uint32)
            sem = raw & 0xFFFF
            if 30 in sem:
                found_person = True
            if 10 in sem:
                found_car = True
    if found_person and found_car:
        print("[OK] Hero-moment classes detected (person + car) in sequence.")
    elif not found_person:
        print("[WARN] No pedestrian (class 30) detected in first 10 frames.")
    elif not found_car:
        print("[WARN] No vehicle (class 10) detected in first 10 frames.")

    print("\nTASK-002 complete. data/demo_sequence/ is ready.")
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
        help="Skip label download (labels already present); just trim velodyne.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip all downloads; assume raw/ is already populated.",
    )
    sys.exit(main(parser.parse_args()))
