"""
test_model_backend.py
======================
Tests for perception/backends/model_backend.py.
"""

from pathlib import Path
import numpy as np
import pytest

from perception.backends.model_backend import (
    classify,
    benchmark_inference,
    extract_point_features,
)
from perception.taxonomy import BASE_CLASSES


def test_feature_extraction():
    """Verify feature extractor computes correct 7-channel geometric representations."""
    synthetic_pts = np.array([
        [10.0, 0.0, 1.0, 0.5],
        [0.0, 20.0, -1.0, 0.8],
        [5.0, 5.0, 0.0, 0.2],
    ], dtype=np.float32)

    feats = extract_point_features(synthetic_pts)
    assert feats.shape == (3, 7)
    assert feats.dtype == np.float32

    # Check range_xy for [10, 0, 1]
    assert np.isclose(feats[0, 4], 10.0)
    # Check dist_3d for [10, 0, 1]
    assert np.isclose(feats[0, 5], np.sqrt(101.0))


def test_model_backend_output_shape_and_types():
    """Verify model backend adheres to the exact (N, 4) and (N,) contract."""
    points, class_ids = classify(0)

    assert isinstance(points, np.ndarray)
    assert isinstance(class_ids, np.ndarray)
    assert points.ndim == 2
    assert points.shape[1] == 4
    assert points.dtype == np.float32

    assert class_ids.ndim == 1
    assert class_ids.shape[0] == points.shape[0]
    assert class_ids.dtype == np.int32

    # Verify returned class IDs are valid taxonomy IDs
    unique_ids = set(np.unique(class_ids))
    valid_ids = set(BASE_CLASSES.keys())
    assert unique_ids.issubset(valid_ids), f"Unknown classes detected: {unique_ids - valid_ids}"


def test_model_benchmark_returns_valid_metrics():
    """Verify inference benchmark execution and returned metrics format."""
    bench = benchmark_inference(num_frames=2)
    assert "avg_latency_ms" in bench
    assert "p95_latency_ms" in bench
    assert "fps" in bench
    assert "avg_points" in bench
    assert bench["avg_latency_ms"] > 0
    assert bench["fps"] > 0
    assert bench["avg_points"] > 0
