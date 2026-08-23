"""
cell.py
=======
Cell data structure for the 2.5D Variable-Resolution Grid Engine.

Each cell represents a spatial voxel / pillar footprint in the 2.5D elevation
and semantic representation.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(slots=True)
class Cell:
    """
    Represents a single cell in the variable-resolution 2.5D grid.

    Attributes
    ----------
    x : float
        Center x coordinate in sensor frame (meters).
    y : float
        Center y coordinate in sensor frame (meters).
    size : float
        Grid cell resolution / edge length (meters), e.g. 0.05, 0.15, 0.50.
    height_mean : float
        Mean elevation (z) of all points falling inside this cell.
    height_max : float
        Maximum elevation (z) of points inside this cell.
    height_min : float
        Minimum elevation (z) of points inside this cell.
    height_var : float
        Variance of elevation (z) inside this cell.
    class_id : int
        Dominant semantic class ID determined by safety-hazard priority voting.
    point_count : int
        Total number of LiDAR returns falling within this cell.
    tier : int
        Ring tier index (0: near-field 5cm, 1: mid-range 15cm, 2: far-field 50cm).
    """
    x: float
    y: float
    size: float
    height_mean: float
    height_max: float
    height_min: float
    height_var: float
    class_id: int
    point_count: int
    tier: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert cell to dictionary for lightweight JSON serialization."""
        return {
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "size": round(self.size, 2),
            "height_mean": round(self.height_mean, 2),
            "height_max": round(self.height_max, 2),
            "height_min": round(self.height_min, 2),
            "height_var": round(self.height_var, 3),
            "class_id": int(self.class_id),
            "point_count": int(self.point_count),
            "tier": int(self.tier),
        }
