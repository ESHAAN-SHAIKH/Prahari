"""
service.py
==========
Perception service wrapper and dynamic backend switch for PRAHARI-Lite.

Provides the single entrypoint `classify_frame(frame_idx)` that downstream
modules (e.g., Grid Engine, API Playback loop) invoke.

The active perception backend can be switched instantly via the
`PERCEPTION_BACKEND` configuration variable or `set_active_backend()`:
    - "groundtruth": Dataset ground-truth remapped fallback (TASK-004)
    - "model": Deep neural point-cloud segmentation model (TASK-005)

Startup logs clearly output which perception backend is active.

Interface contract:
-------------------
    classify_frame(frame_idx: int) -> tuple[np.ndarray, np.ndarray]
        - points: np.ndarray, shape (N, 4), dtype np.float32 (x, y, z, intensity)
        - class_ids: np.ndarray, shape (N,), dtype np.int32
"""

import logging
import os
from pathlib import Path
from typing import Literal, Optional, Tuple
import numpy as np

from perception.backends import groundtruth_backend, model_backend

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PRAHARI-Perception] %(message)s",
)
logger = logging.getLogger("perception.service")

# ---------------------------------------------------------------------------
# Backend Selection Flag (Single source of truth)
# ---------------------------------------------------------------------------
# Set to "groundtruth" or "model". Can also be overridden via environment variable.
BackendType = Literal["groundtruth", "model"]
PERCEPTION_BACKEND: BackendType = os.environ.get("PERCEPTION_BACKEND", "groundtruth")  # type: ignore


def log_active_backend(backend: str) -> None:
    """Log the currently active backend with clear formatting."""
    print("=" * 60)
    print(f"[PRAHARI-Lite] Active Perception Backend: '{backend.upper()}'")
    if backend == "groundtruth":
        print("  Mode: Dataset ground-truth annotations + simulated latency fallback")
    elif backend == "model":
        print("  Mode: Real-time neural point-cloud segmentation model inference")
    print("=" * 60)
    logger.info("Active perception backend set to: %s", backend)


# Log on initial module import
log_active_backend(PERCEPTION_BACKEND)


def get_active_backend() -> str:
    """Return the name of the currently active perception backend."""
    global PERCEPTION_BACKEND
    return PERCEPTION_BACKEND


def set_active_backend(backend: BackendType) -> None:
    """
    Dynamically change the active perception backend.

    Parameters
    ----------
    backend : Literal["groundtruth", "model"]
        The backend to activate.
    """
    global PERCEPTION_BACKEND
    if backend not in ("groundtruth", "model"):
        raise ValueError(f"Invalid backend '{backend}'. Supported: 'groundtruth', 'model'")
    PERCEPTION_BACKEND = backend
    log_active_backend(backend)


# ---------------------------------------------------------------------------
# Primary Perception Service Entrypoint
# ---------------------------------------------------------------------------

def classify_frame(
    frame_idx: int,
    sequence_dir: Optional[Path] = None,
    backend: Optional[BackendType] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify points for a given LiDAR frame using the selected perception backend.

    Parameters
    ----------
    frame_idx : int
        Zero-based index of the LiDAR frame.
    sequence_dir : Path, optional
        Custom directory containing the demo sequence data.
    backend : str, optional
        Override the default backend for this single invocation.

    Returns
    -------
    points : np.ndarray, shape (N, 4), dtype np.float32
        Point coordinates and reflectance [x, y, z, intensity].
    class_ids : np.ndarray, shape (N,), dtype np.int32
        Per-point classified base class IDs.
    """
    target_backend = backend or PERCEPTION_BACKEND

    if target_backend == "groundtruth":
        return groundtruth_backend.classify(frame_idx, sequence_dir=sequence_dir)
    elif target_backend == "model":
        return model_backend.classify(frame_idx, sequence_dir=sequence_dir)
    else:
        raise ValueError(
            f"Unknown perception backend: '{target_backend}'. Must be 'groundtruth' or 'model'."
        )
