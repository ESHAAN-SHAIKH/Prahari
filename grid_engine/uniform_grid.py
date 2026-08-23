"""
uniform_grid.py
===============
Baseline Uniform High-Resolution 2.5D Elevation & Semantic Grid Engine.

Maintains a fixed, uniform 5 cm (0.05 m) cell size across the entire
sensing radius (0 - 100 m).

This serves as the naive high-resolution baseline against which the
variable-resolution RingGrid is benchmarked to compute memory reduction
and efficiency metrics (TASK-009, TASK-017).
"""

from typing import Dict, Optional, Tuple
import numpy as np

from grid_engine.cell import Cell

PRIORITY_RANK: Dict[int, int] = {
    2: 5,    # dynamic_pedestrian (Highest priority)
    3: 4,    # dynamic_vehicle
    1: 3,    # static_obstacle
    0: 2,    # terrain_drivable
    255: 1,  # unknown (Lowest priority)
}

RANK_TO_CLASS: Dict[int, int] = {
    5: 2,
    4: 3,
    3: 1,
    2: 0,
    1: 255,
}


class UniformGrid:
    """
    Uniform High-Resolution (5cm) 2.5D Grid Engine.
    """

    def __init__(self, cell_size_m: float = 0.05):
        self.cell_size_m = float(cell_size_m)
        self.priority_lut = np.array([PRIORITY_RANK.get(c, 1) for c in range(256)], dtype=np.int32)
        self.rank_to_class_lut = np.array([RANK_TO_CLASS.get(r, 255) for r in range(6)], dtype=np.int32)

    def project(
        self,
        points: np.ndarray,
        class_ids: np.ndarray,
        max_range: float = 100.0,
    ) -> Dict[Tuple[int, int], Cell]:
        """
        Project 3D points onto a uniform 5cm 2.5D grid.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3) or (N, 4)
            Point coordinates [x, y, z, ...].
        class_ids : np.ndarray, shape (N,)
            Semantic class IDs per point.
        max_range : float, default 100.0
            Maximum range radius in meters.

        Returns
        -------
        grid_cells : Dict[Tuple[ix, iy], Cell]
            Active cells indexed by (ix, iy).
        """
        if len(points) == 0:
            return {}

        x = points[:, 0].astype(np.float32)
        y = points[:, 1].astype(np.float32)
        z = points[:, 2].astype(np.float32)

        r = np.sqrt(x ** 2 + y ** 2)
        valid_mask = (r <= max_range)
        if not np.any(valid_mask):
            return {}

        x = x[valid_mask]
        y = y[valid_mask]
        z = z[valid_mask]
        class_ids = class_ids[valid_mask]

        # Uniform 5cm coordinate indexing
        ix = np.floor(x / self.cell_size_m).astype(np.int64)
        iy = np.floor(y / self.cell_size_m).astype(np.int64)

        OFFSET = 1 << 27
        packed_keys = ((ix + OFFSET) << 28) | (iy + OFFSET)

        unique_keys, inverse_idx, point_counts = np.unique(
            packed_keys,
            return_inverse=True,
            return_counts=True,
        )

        num_cells = len(unique_keys)

        # Height statistics
        sum_z = np.bincount(inverse_idx, weights=z, minlength=num_cells)
        mean_z = (sum_z / point_counts).astype(np.float32)

        sum_z2 = np.bincount(inverse_idx, weights=(z ** 2), minlength=num_cells)
        var_z = np.maximum(0.0, (sum_z2 / point_counts) - (mean_z ** 2)).astype(np.float32)

        min_z = np.full(num_cells, np.inf, dtype=np.float32)
        np.minimum.at(min_z, inverse_idx, z)

        max_z = np.full(num_cells, -np.inf, dtype=np.float32)
        np.maximum.at(max_z, inverse_idx, z)

        # Class priority voting
        pt_priorities = self.priority_lut[np.clip(class_ids, 0, 255)]
        cell_max_priorities = np.zeros(num_cells, dtype=np.int32)
        np.maximum.at(cell_max_priorities, inverse_idx, pt_priorities)
        cell_classes = self.rank_to_class_lut[cell_max_priorities]

        u_ix = ((unique_keys >> 28) & 0x0FFFFFFF).astype(np.int64) - OFFSET
        u_iy = (unique_keys & 0x0FFFFFFF).astype(np.int64) - OFFSET

        u_cx = (u_ix + 0.5) * self.cell_size_m
        u_cy = (u_iy + 0.5) * self.cell_size_m

        grid_dict: Dict[Tuple[int, int], Cell] = {}
        for i in range(num_cells):
            c_ix = int(u_ix[i])
            c_iy = int(u_iy[i])
            grid_dict[(c_ix, c_iy)] = Cell(
                x=float(u_cx[i]),
                y=float(u_cy[i]),
                size=float(self.cell_size_m),
                height_mean=float(mean_z[i]),
                height_max=float(max_z[i]),
                height_min=float(min_z[i]),
                height_var=float(var_z[i]),
                class_id=int(cell_classes[i]),
                point_count=int(point_counts[i]),
                tier=0,
            )

        return grid_dict
