"""
test_service.py
===============
Tests for perception/service.py: backend routing, dynamic switching, and contract validation.
"""

from pathlib import Path
import numpy as np
import pytest

from perception.service import (
    classify_frame,
    get_active_backend,
    set_active_backend,
    PERCEPTION_BACKEND,
)
from perception.taxonomy import BASE_CLASSES


def test_service_dispatches_to_groundtruth_backend():
    """Verify service dispatches to groundtruth backend when active."""
    set_active_backend("groundtruth")
    assert get_active_backend() == "groundtruth"

    pts, classes = classify_frame(0)
    assert isinstance(pts, np.ndarray)
    assert isinstance(classes, np.ndarray)
    assert pts.shape[1] == 4
    assert pts.dtype == np.float32
    assert classes.dtype == np.int32
    assert pts.shape[0] == classes.shape[0]


def test_service_dispatches_to_model_backend():
    """Verify service dispatches to neural model backend when active."""
    set_active_backend("model")
    assert get_active_backend() == "model"

    pts, classes = classify_frame(0)
    assert isinstance(pts, np.ndarray)
    assert isinstance(classes, np.ndarray)
    assert pts.shape[1] == 4
    assert pts.dtype == np.float32
    assert classes.dtype == np.int32
    assert pts.shape[0] == classes.shape[0]


def test_service_per_call_backend_override():
    """Verify backend argument overrides global PERCEPTION_BACKEND per call."""
    set_active_backend("groundtruth")

    # Explicitly request model backend without mutating global state
    pts, classes = classify_frame(0, backend="model")
    assert get_active_backend() == "groundtruth"
    assert pts.shape[0] == classes.shape[0]


def test_invalid_backend_raises_error():
    """Verify invalid backend names raise ValueError."""
    with pytest.raises(ValueError):
        set_active_backend("invalid_backend_name")  # type: ignore

    with pytest.raises(ValueError):
        classify_frame(0, backend="non_existent")  # type: ignore
