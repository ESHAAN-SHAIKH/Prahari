"""
test_taxonomy.py
================
Tests for perception/taxonomy.py: label remapping, completeness, and priority.
"""

from pathlib import Path
import numpy as np
import pytest

from perception.taxonomy import (
    BASE_CLASSES,
    BASE_CLASS_IDS,
    BASE_CLASS_COLORS,
    CLASS_PRIORITY,
    SEMANTICKITTI_TO_BASE,
    map_labels,
    validate_completeness,
)


def test_taxonomy_completeness_on_sequence():
    """Ensure no raw class in the sequence is left unmapped."""
    unmapped = validate_completeness()
    assert not unmapped, f"Found unmapped SemanticKITTI classes: {unmapped}"


def test_map_labels_mapping():
    """Test known SemanticKITTI IDs map to correct BASE_CLASSES."""
    # Synthetic array with specific SemanticKITTI IDs (combined with arbitrary instance ID in high 16 bits)
    raw_samples = np.array([
        40 | (1 << 16),   # road -> terrain_drivable (0)
        44,               # parking -> terrain_drivable (0)
        50,               # building -> static_obstacle (1)
        71,               # pole -> static_obstacle (1)
        30 | (5 << 16),   # person -> dynamic_pedestrian (2)
        10,               # car -> dynamic_vehicle (3)
        0,                # unlabeled -> unknown (255)
        9999,             # unmapped -> unknown (255)
    ], dtype=np.uint32)

    mapped = map_labels(raw_samples)
    expected = np.array([0, 0, 1, 1, 2, 3, 255, 255], dtype=np.int32)

    np.testing.assert_array_equal(mapped, expected)


def test_taxonomy_dictionaries_integrity():
    """Verify data structures, colors, and priority mappings are coherent."""
    # Every base class must have a color
    for class_id in BASE_CLASSES:
        assert class_id in BASE_CLASS_COLORS
        color = BASE_CLASS_COLORS[class_id]
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    # Every base class must have a defined priority
    for class_id in BASE_CLASSES:
        assert class_id in CLASS_PRIORITY

    # Dynamic hazard classes should outrank terrain and static obstacle
    assert CLASS_PRIORITY[BASE_CLASS_IDS["dynamic_pedestrian"]] > CLASS_PRIORITY[BASE_CLASS_IDS["terrain_drivable"]]
    assert CLASS_PRIORITY[BASE_CLASS_IDS["dynamic_vehicle"]] > CLASS_PRIORITY[BASE_CLASS_IDS["terrain_drivable"]]
    assert CLASS_PRIORITY[BASE_CLASS_IDS["static_obstacle"]] > CLASS_PRIORITY[BASE_CLASS_IDS["terrain_drivable"]]
