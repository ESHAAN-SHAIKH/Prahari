"""
test_loader.py
==============
Tests for perception/loader.py: load_frame and get_classified_points.
"""

from pathlib import Path
import numpy as np
import pytest

from perception.loader import load_frame, get_classified_points, get_frame_count
from perception.taxonomy import BASE_CLASSES


def test_loader_shapes():
    """Verify points are (N, 4) float32 and labels are (N,) uint32 with matching N."""
    frame_count = get_frame_count()
    assert frame_count > 0, "No frames available in data/demo_sequence"

    points, raw_labels = load_frame(0)
    assert isinstance(points, np.ndarray)
    assert isinstance(raw_labels, np.ndarray)
    assert points.ndim == 2
    assert points.shape[1] == 4
    assert points.dtype == np.float32
    assert raw_labels.ndim == 1
    assert raw_labels.dtype == np.uint32
    assert points.shape[0] == raw_labels.shape[0]


def test_get_classified_points_contract():
    """Verify get_classified_points adheres to the frozen interface contract."""
    points, class_ids = get_classified_points(0)
    
    assert isinstance(points, np.ndarray)
    assert isinstance(class_ids, np.ndarray)
    assert points.ndim == 2
    assert points.shape[1] == 4
    assert points.dtype == np.float32
    assert class_ids.ndim == 1
    assert class_ids.dtype == np.int32
    assert points.shape[0] == class_ids.shape[0]

    # Verify that all class_ids belong to the valid BASE_CLASSES taxonomy
    unique_classes = set(np.unique(class_ids))
    valid_classes = set(BASE_CLASSES.keys())
    invalid = unique_classes - valid_classes
    assert not invalid, f"Encountered unexpected class IDs: {invalid}"


def test_invalid_frame_index_raises_error():
    """Attempting to load a non-existent frame should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_frame(999999)
