"""
ring_grid.py
============
Distance-Tiered Variable-Resolution 2.5D Elevation & Semantic Grid Engine
with Risk-Adaptive Local Subdivision.

Implements concentric ring-based spatial decomposition:
    - Tier 0 (Near field:  0.0m - 10.0m) : 0.05m (5cm)  resolution
    - Tier 1 (Mid range:  10.0m - 30.0m) : 0.15m (15cm) resolution
    - Tier 2 (Far field:  30.0m - 100.0m): 0.50m (50cm) resolution

⭐ Risk-Adaptive Rule (TASK-010):
----------------------------------
When `enable_risk_adaptive=True`, coarse cells exhibiting high height variance
(e.g., potholes, obstacles, steep terrain roughness) are dynamically subdivided
into finer sub-cells (e.g. 15cm -> 5cm or 50cm -> 15cm), prioritizing safety
regardless of distance from the sensor.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from grid_engine.cell import Cell
from perception.taxonomy import BASE_CLASSES

# ---------------------------------------------------------------------------
# Ring Tier Configuration (Matches Problem Statement Specification)
# ---------------------------------------------------------------------------
DEFAULT_RING_TIERS: List[Dict[str, float]] = [
    {"tier": 0, "max_range_m": 10.0, "cell_size_m": 0.05},   # 5cm near-field
    {"tier": 1, "max_range_m": 30.0, "cell_size_m": 0.15},   # 15cm mid-range
    {"tier": 2, "max_range_m": 100.0, "cell_size_m": 0.50},  # 50cm far-field
]

# Priority rank mapping: higher value wins cell classification
PRIORITY_RANK: Dict[int, int] = {
    2: 5,    # dynamic_pedestrian (Highest priority hazard)
    3: 4,    # dynamic_vehicle
    1: 3,    # static_obstacle
    0: 2,    # terrain_drivable
    255: 1,  # unknown (Lowest priority)
}

# Reverse mapping: Priority Rank -> Base Class ID
RANK_TO_CLASS: Dict[int, int] = {
    5: 2,    # pedestrian
    4: 3,    # vehicle
    3: 1,    # obstacle
    2: 0,    # terrain
    1: 255,  # unknown
}


class RingGrid:
    """
    Adaptive Variable-Resolution 2.5D Ring Grid Engine with Risk-Adaptive Subdivision.
    """

    def __init__(self, tiers: Optional[List[Dict[str, float]]] = None):
        self.tiers = tiers or DEFAULT_RING_TIERS
        # Sort tiers by ascending max_range_m
        self.tiers = sorted(self.tiers, key=lambda t: t["max_range_m"])
        self.num_tiers = len(self.tiers)
        self.max_ranges = np.array([t["max_range_m"] for t in self.tiers], dtype=np.float64)
        self.cell_sizes = np.array([t["cell_size_m"] for t in self.tiers], dtype=np.float64)

        # Precompute lookup tables
        self.priority_lut = np.array([PRIORITY_RANK.get(c, 1) for c in range(256)], dtype=np.int32)
        self.rank_to_class_lut = np.array([RANK_TO_CLASS.get(r, 255) for r in range(6)], dtype=np.int32)

    def assign_tiers(self, range_xy: np.ndarray) -> np.ndarray:
        """
        Assign each point to a resolution tier based on 2D range r = sqrt(x^2 + y^2).
        
        Boundary rule: A point at exactly max_range_m belongs to the INNER tier.
        (e.g., r=10.0m -> Tier 0, r=10.0001m -> Tier 1).
        
        Parameters
        ----------
        range_xy : np.ndarray, shape (N,)
            Radial planar distances in meters.

        Returns
        -------
        tier_indices : np.ndarray, shape (N,), dtype int32
        """
        tier_indices = np.full(len(range_xy), self.num_tiers - 1, dtype=np.int32)

        # Outer-to-inner assignment ensures r <= max_range_m receives the inner tier
        for tier_idx in range(self.num_tiers - 1, -1, -1):
            max_r = self.max_ranges[tier_idx]
            mask = (range_xy <= max_r)
            tier_indices[mask] = tier_idx

        return tier_indices

    def project(
        self,
        points: np.ndarray,
        class_ids: np.ndarray,
        max_range: float = 100.0,
        enable_risk_adaptive: bool = False,
        variance_threshold: float = 0.04,
    ) -> Dict[Tuple[int, int, int], Cell]:
        """
        Project 3D point cloud into variable-resolution 2.5D grid cells.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3) or (N, 4)
            Point cloud array with [x, y, z, ...].
        class_ids : np.ndarray, shape (N,)
            Semantic class IDs per point.
        max_range : float, default 100.0
            Maximum range threshold in meters (points beyond are excluded).
        enable_risk_adaptive : bool, default False
            If True, coarse cells with height variance exceeding variance_threshold
            are subdivided into finer-resolution sub-cells.
        variance_threshold : float, default 0.04
            Height variance threshold (m^2) triggering local risk subdivision.

        Returns
        -------
        grid_cells : Dict[Tuple[tier, ix, iy], Cell]
            Dictionary of active spatial cells mapped by (tier, ix, iy).
        """
        if len(points) == 0:
            return {}

        x = points[:, 0].astype(np.float32)
        y = points[:, 1].astype(np.float32)
        z = points[:, 2].astype(np.float32)

        # 1. Planar range computation
        r = np.sqrt(x ** 2 + y ** 2)

        # Range filter
        valid_mask = (r <= max_range)
        if not np.any(valid_mask):
            return {}

        x = x[valid_mask]
        y = y[valid_mask]
        z = z[valid_mask]
        r = r[valid_mask]
        class_ids = class_ids[valid_mask]

        # 2. Tier Assignment (Vectorized Base Pass)
        point_tiers = self.assign_tiers(r)

        OFFSET = 1 << 27

        # Helper to compute packed keys and unique cells
        def compute_grouping(tiers_arr):
            point_sizes = self.cell_sizes[tiers_arr]
            ix_arr = np.floor(x / point_sizes).astype(np.int64)
            iy_arr = np.floor(y / point_sizes).astype(np.int64)
            packed = (
                (tiers_arr.astype(np.int64) << 56) |
                ((ix_arr + OFFSET) << 28) |
                (iy_arr + OFFSET)
            )
            return np.unique(packed, return_inverse=True, return_counts=True)

        unique_keys, inverse_idx, point_counts = compute_grouping(point_tiers)
        num_cells = len(unique_keys)

        # 3. Risk-Adaptive Subdivision Pass (TASK-010)
        if enable_risk_adaptive:
            # Check height variance per cell
            sum_z = np.bincount(inverse_idx, weights=z, minlength=num_cells)
            mean_z = (sum_z / point_counts).astype(np.float32)
            sum_z2 = np.bincount(inverse_idx, weights=(z ** 2), minlength=num_cells)
            var_z = np.maximum(0.0, (sum_z2 / point_counts) - (mean_z ** 2)).astype(np.float32)

            u_tiers_tmp = (unique_keys >> 56).astype(np.int32)
            # Find coarse cells (tier > 0) where variance exceeds threshold
            subdivide_cells = np.where((var_z >= variance_threshold) & (u_tiers_tmp > 0) & (point_counts >= 3))[0]

            if len(subdivide_cells) > 0:
                subdivide_mask = np.isin(inverse_idx, subdivide_cells)
                # Refine to Tier 0 (or next finer tier)
                point_tiers[subdivide_mask] = np.maximum(0, point_tiers[subdivide_mask] - 1)

                # Re-compute unique cell grouping with subdivided tiers
                unique_keys, inverse_idx, point_counts = compute_grouping(point_tiers)
                num_cells = len(unique_keys)

        # 4. Vectorized Height and Statistical Aggregations
        sum_z = np.bincount(inverse_idx, weights=z, minlength=num_cells)
        mean_z = (sum_z / point_counts).astype(np.float32)

        sum_z2 = np.bincount(inverse_idx, weights=(z ** 2), minlength=num_cells)
        var_z = np.maximum(0.0, (sum_z2 / point_counts) - (mean_z ** 2)).astype(np.float32)

        min_z = np.full(num_cells, np.inf, dtype=np.float32)
        np.minimum.at(min_z, inverse_idx, z)

        max_z = np.full(num_cells, -np.inf, dtype=np.float32)
        np.maximum.at(max_z, inverse_idx, z)

        # 5. Hazard Priority Majority Class Voting
        pt_priorities = self.priority_lut[np.clip(class_ids, 0, 255)]
        cell_max_priorities = np.zeros(num_cells, dtype=np.int32)
        np.maximum.at(cell_max_priorities, inverse_idx, pt_priorities)
        cell_classes = self.rank_to_class_lut[cell_max_priorities]

        # 6. Unpack coordinates & centers
        u_tiers = (unique_keys >> 56).astype(np.int32)
        u_ix = ((unique_keys >> 28) & 0x0FFFFFFF).astype(np.int64) - OFFSET
        u_iy = (unique_keys & 0x0FFFFFFF).astype(np.int64) - OFFSET

        u_sizes = self.cell_sizes[u_tiers]
        u_cx = (u_ix + 0.5) * u_sizes
        u_cy = (u_iy + 0.5) * u_sizes

        # 7. Fast Dictionary Assembly
        grid_dict: Dict[Tuple[int, int, int], Cell] = {}
        for i in range(num_cells):
            t = int(u_tiers[i])
            c_ix = int(u_ix[i])
            c_iy = int(u_iy[i])
            grid_dict[(t, c_ix, c_iy)] = Cell(
                x=float(u_cx[i]),
                y=float(u_cy[i]),
                size=float(u_sizes[i]),
                height_mean=float(mean_z[i]),
                height_max=float(max_z[i]),
                height_min=float(min_z[i]),
                height_var=float(var_z[i]),
                class_id=int(cell_classes[i]),
                point_count=int(point_counts[i]),
                tier=t,
            )

        return grid_dict
