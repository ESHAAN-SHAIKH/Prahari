"""
loader.py
=========
Unified point loader for PRAHARI-Lite.

This is the single interface contract for reading LiDAR point clouds and
their semantic classifications. Downstream components (Perception Backends,
Grid Engine, Playback Service) import from here.

Interface contract:
-------------------
    get_classified_points(frame_idx: int) -> tuple[np.ndarray, np.ndarray]
        - points: np.ndarray, shape (N, 4), dtype np.float32  (x, y, z, intensity)
        - base_class_ids: np.ndarray, shape (N,), dtype np.int32
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from perception.taxonomy import map_labels

# Default path relative to repository root
DEFAULT_DEMO_SEQUENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "demo_sequence"


def get_sequence_paths(sequence_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Return (velodyne_dir, labels_dir) for the given sequence root."""
    base = Path(sequence_dir) if sequence_dir is not None else DEFAULT_DEMO_SEQUENCE_DIR
    return base / "velodyne", base / "labels"


def get_frame_count(sequence_dir: Optional[Path] = None) -> int:
    """Return the total number of frames in the demo sequence."""
    vel_dir, _ = get_sequence_paths(sequence_dir)
    if not vel_dir.exists():
        return 0
    return len(list(vel_dir.glob("*.bin")))


def load_frame(
    frame_idx: int,
    sequence_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load raw points and raw labels for a single frame index.

    Parameters
    ----------
    frame_idx : int
        Zero-based frame index.
    sequence_dir : Path, optional
        Path to sequence directory containing velodyne/ and labels/ subfolders.

    Returns
    -------
    points : np.ndarray, shape (N, 4), dtype np.float32
        Points formatted as [x, y, z, intensity].
    raw_labels : np.ndarray, shape (N,), dtype np.uint32
        Raw SemanticKITTI uint32 labels.
    """
    vel_dir, lbl_dir = get_sequence_paths(sequence_dir)
    
    bin_path = vel_dir / f"{frame_idx:06d}.bin"
    lbl_path = lbl_dir / f"{frame_idx:06d}.label"

    if not bin_path.exists():
        raise FileNotFoundError(f"Velodyne binary file not found: {bin_path}")
    if not lbl_path.exists():
        raise FileNotFoundError(f"Label file not found: {lbl_path}")

    # Read binary points: float32, [x, y, z, intensity]
    raw_points = np.fromfile(bin_path, dtype=np.float32)
    if raw_points.size % 4 != 0:
        raise ValueError(f"Corrupt .bin file {bin_path}: size not divisible by 4 floats.")
    points = raw_points.reshape(-1, 4)

    # Read raw labels: uint32 per point
    raw_labels = np.fromfile(lbl_path, dtype=np.uint32)

    if points.shape[0] != raw_labels.shape[0]:
        raise ValueError(
            f"Point count mismatch in frame {frame_idx}: "
            f"{points.shape[0]} points vs {raw_labels.shape[0]} labels."
        )

    return points, raw_labels


def get_classified_points(
    frame_idx: int,
    sequence_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Primary interface contract for PRAHARI-Lite.
    
    Loads a frame and returns points with mapped base class IDs.

    Parameters
    ----------
    frame_idx : int
        Zero-based frame index.
    sequence_dir : Path, optional
        Path to sequence root.

    Returns
    -------
    points : np.ndarray, shape (N, 4), dtype np.float32
        Point coordinates and intensity [x, y, z, intensity].
    class_ids : np.ndarray, shape (N,), dtype np.int32
        Mapped base class IDs (from perception.taxonomy.BASE_CLASSES).
    """
    points, raw_labels = load_frame(frame_idx, sequence_dir=sequence_dir)
    class_ids = map_labels(raw_labels)
    return points, class_ids
