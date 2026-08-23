"""
test_groundtruth_backend.py
===========================
Tests for perception/backends/groundtruth_backend.py.
"""

import time
import numpy as np
import pytest

from perception.backends.groundtruth_backend import (
    classify,
    SIMULATED_INFERENCE_DELAY_S,
)
from perception.loader import get_classified_points


def test_groundtruth_backend_matches_loader_output():
    """Verify ground-truth backend output matches get_classified_points exactly."""
    points_gt, classes_gt = classify(0, simulate_delay=False)
    points_loader, classes_loader = get_classified_points(0)

    assert isinstance(points_gt, np.ndarray)
    assert isinstance(classes_gt, np.ndarray)
    assert points_gt.shape == points_loader.shape
    assert classes_gt.shape == classes_loader.shape
    assert points_gt.dtype == points_loader.dtype
    assert classes_gt.dtype == classes_loader.dtype

    np.testing.assert_array_equal(points_gt, points_loader)
    np.testing.assert_array_equal(classes_gt, classes_loader)


def test_groundtruth_backend_simulates_inference_delay():
    """Verify simulated delay is executed when simulate_delay=True."""
    start_time = time.perf_counter()
    _ = classify(0, simulate_delay=True)
    elapsed = time.perf_counter() - start_time

    # Elapsed time must be at least the configured simulated delay
    assert elapsed >= SIMULATED_INFERENCE_DELAY_S * 0.9, (
        f"Elapsed {elapsed:.4f}s is less than simulated delay {SIMULATED_INFERENCE_DELAY_S:.4f}s"
    )
