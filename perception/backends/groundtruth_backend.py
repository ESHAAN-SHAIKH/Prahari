"""
groundtruth_backend.py
======================
Ground-truth relabeling perception backend for PRAHARI-Lite.

This backend serves as the primary fallback perception pipeline. It utilizes
the ground-truth semantic annotations provided by the dataset, remapping them
into the unified 5-class taxonomy defined in `perception.taxonomy`.

An artificial simulated inference delay (default ~8 ms) is incorporated to
simulate realistic laptop/edge inference latency in the downstream metrics
instrumentation panel (TASK-014), avoiding misleading 0 ms readings.

Interface contract:
-------------------
    classify(frame_idx: int) -> tuple[np.ndarray, np.ndarray]
        - points: np.ndarray, shape (N, 4), dtype np.float32 (x, y, z, intensity)
        - class_ids: np.ndarray, shape (N,), dtype np.int32
"""

import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from perception.loader import get_classified_points

# Placeholder inference delay in seconds (e.g., 8 ms).
# Explicitly labeled per blueprint specifications as a simulated placeholder
# for realistic LiDAR segmentation model execution time.
SIMULATED_INFERENCE_DELAY_S: float = 0.008


def classify(
    frame_idx: int,
    sequence_dir: Optional[Path] = None,
    simulate_delay: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform ground-truth perception classification for a frame.

    Parameters
    ----------
    frame_idx : int
        Index of the frame to classify.
    sequence_dir : Path, optional
        Custom directory containing the demo sequence data.
    simulate_delay : bool, default True
        Whether to sleep for SIMULATED_INFERENCE_DELAY_S to simulate
        realistic inference latency.

    Returns
    -------
    points : np.ndarray, shape (N, 4), dtype np.float32
        Point cloud coordinates and reflectance [x, y, z, intensity].
    class_ids : np.ndarray, shape (N,), dtype np.int32
        Per-point classified base class IDs.
    """
    if simulate_delay and SIMULATED_INFERENCE_DELAY_S > 0:
        time.sleep(SIMULATED_INFERENCE_DELAY_S)

    points, class_ids = get_classified_points(frame_idx, sequence_dir=sequence_dir)
    return points, class_ids
