"""
serialize.py
============
Grid serialization utility for HTTP API and WebSocket transport to the dashboard.

Transforms the internal grid dictionary into a compact, lightweight JSON payload
containing only the essential fields needed for real-time frontend rendering:
    - x, y: 2D cell center coordinates (rounded to 2 decimal places)
    - size: Cell spatial resolution (0.05, 0.15, 0.50)
    - height: Elevation mean for pseudo-2.5D extrusion
    - class_id: Hazard-voted semantic category (0: terrain, 1: obstacle, 2: person, 3: vehicle)
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from grid_engine.cell import Cell


def grid_to_dict(
    grid: Dict[Any, Cell],
    frame_idx: Optional[int] = None,
    include_stats: bool = False,
) -> Dict[str, Any]:
    """
    Convert a grid dictionary into a clean dictionary payload for API transport.

    Parameters
    ----------
    grid : Dict[Any, Cell]
        Active cells dictionary returned by RingGrid or UniformGrid.
    frame_idx : int, optional
        LiDAR frame sequence index.
    include_stats : bool, default False
        Whether to include extended variance and min/max elevation fields.

    Returns
    -------
    Dict[str, Any]
        Structured payload with cell list and metadata.
    """
    cells_list: List[Dict[str, Any]] = []

    if include_stats:
        for cell in grid.values():
            cells_list.append({
                "x": round(cell.x, 2),
                "y": round(cell.y, 2),
                "size": round(cell.size, 2),
                "height": round(cell.height_mean, 2),
                "height_max": round(cell.height_max, 2),
                "height_min": round(cell.height_min, 2),
                "height_var": round(cell.height_var, 3),
                "class_id": cell.class_id,
                "point_count": cell.point_count,
                "tier": cell.tier,
            })
    else:
        for cell in grid.values():
            cells_list.append({
                "x": round(cell.x, 2),
                "y": round(cell.y, 2),
                "size": round(cell.size, 2),
                "height": round(cell.height_mean, 2),
                "class_id": cell.class_id,
            })

    payload: Dict[str, Any] = {
        "frame_idx": frame_idx,
        "cell_count": len(cells_list),
        "cells": cells_list,
    }

    return payload


def grid_to_json(
    grid: Dict[Any, Cell],
    frame_idx: Optional[int] = None,
    include_stats: bool = False,
) -> str:
    """
    Serialize a grid dictionary to a JSON string.

    Parameters
    ----------
    grid : Dict[Any, Cell]
        Active cells dictionary.
    frame_idx : int, optional
        LiDAR frame index.
    include_stats : bool, default False
        Whether to include extended variance and min/max stats.

    Returns
    -------
    str
        JSON formatted payload.
    """
    dict_payload = grid_to_dict(grid, frame_idx=frame_idx, include_stats=include_stats)
    return json.dumps(dict_payload, separators=(',', ':'))


def benchmark_serialization(
    grid: Dict[Any, Cell],
    num_trials: int = 10,
) -> Dict[str, Union[float, int]]:
    """
    Benchmark grid serialization performance on a computed frame.

    Returns
    -------
    Dict with metrics:
        - "cell_count": int
        - "avg_serialize_ms": float
        - "p95_serialize_ms": float
        - "payload_kb": float
        - "payload_mb": float
    """
    latencies: List[float] = []
    payload_str = ""

    for _ in range(num_trials):
        t0 = time.perf_counter()
        payload_str = grid_to_json(grid)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

    payload_bytes = len(payload_str.encode("utf-8"))

    return {
        "cell_count": len(grid),
        "avg_serialize_ms": round(float(sum(latencies) / len(latencies)), 2),
        "p95_serialize_ms": round(float(sorted(latencies)[int(len(latencies) * 0.95)]), 2),
        "payload_kb": round(payload_bytes / 1024.0, 2),
        "payload_mb": round(payload_bytes / (1024.0 * 1024.0), 3),
    }
