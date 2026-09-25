"""Variable-resolution 2.5D grid engine.

Projects classified 3D points into a quadtree whose cell size is driven by range **and** by
risk: hazard class, sensing confidence, and class-boundary complexity. Cells are 2.5D — each
carries elevation statistics alongside its semantic label, so curbs, potholes and overhanging
structure survive the 3D -> 2.5D projection that a flat occupancy grid throws away.

**Alignment**, which the problem statement calls out as the hard part: points are quantised
once onto the finest grid (5 cm) and sorted by Morton code. Every quadtree node is then a
contiguous run of that sorted array, so a point belongs to exactly one cell at every level and
cannot fall between two cells of differing resolution. Alignment errors and data loss at
resolution boundaries are structurally impossible here rather than detected and repaired.

**Performance**: the build is two passes. The descent computes only O(1) prefix-sum lookups
and integer bisections per node, collecting leaf boundaries; aggregation then runs vectorised
over all leaves at once. That is what keeps it inside a 10 Hz budget in pure Python.
"""
from __future__ import annotations

import time

from dataclasses import dataclass, field

import numpy as np

from . import config as C


@dataclass
class Grid:
    """One built frame, as struct-of-arrays."""
    x: np.ndarray           # cell centre x (m)
    y: np.ndarray           # cell centre y (m)
    level: np.ndarray       # quadtree level; size = ROOT_SIZE_M / 2**level
    z: np.ndarray           # mean elevation (m) — the 2.5D payload
    z_min: np.ndarray
    z_max: np.ndarray
    cls: np.ndarray         # dominant class id
    conf: np.ndarray        # mean sensing confidence, 0..1
    driver: np.ndarray      # which rule set this cell's resolution
    n_points: np.ndarray
    code: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.x.size)

    @property
    def size(self) -> np.ndarray:
        return C.ROOT_SIZE_M / np.power(2.0, self.level.astype(np.float64))


def _morton(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Interleave two 12-bit integers into a 24-bit Morton code, x in the high bit of each
    pair. Sorting by this code puts every quadtree node's points in one contiguous run."""
    def spread(v):
        v = v.astype(np.uint32) & np.uint32(0x0000FFFF)
        v = (v | (v << np.uint32(8))) & np.uint32(0x00FF00FF)
        v = (v | (v << np.uint32(4))) & np.uint32(0x0F0F0F0F)
        v = (v | (v << np.uint32(2))) & np.uint32(0x33333333)
        v = (v | (v << np.uint32(1))) & np.uint32(0x55555555)
        return v
    return ((spread(ix) << np.uint32(1)) | spread(iy)).astype(np.int64)


def _demorton(code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of _morton: recover (ix, iy) from an interleaved prefix."""
    def compact(v):
        v = v.astype(np.uint64) & np.uint64(0x5555555555555555)
        v = (v | (v >> np.uint64(1))) & np.uint64(0x3333333333333333)
        v = (v | (v >> np.uint64(2))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        v = (v | (v >> np.uint64(4))) & np.uint64(0x00FF00FF00FF00FF)
        v = (v | (v >> np.uint64(8))) & np.uint64(0x0000FFFF0000FFFF)
        return v.astype(np.int64)
    c = code.astype(np.uint64)
    return compact(c >> np.uint64(1)), compact(c)


class GridEngine:
    def __init__(self,
                 hazard_factor: float = C.HAZARD_REFINE_FACTOR,
                 lowconf_factor: float = C.LOWCONF_REFINE_FACTOR,
                 conf_threshold: float = C.CONF_THRESHOLD,
                 boundary_min_size_m: float = C.BOUNDARY_MIN_SIZE_M,
                 risk_adaptive: bool = True):
        self.hazard_factor = hazard_factor
        self.lowconf_factor = lowconf_factor
        self.conf_threshold = conf_threshold
        self.boundary_min_size_m = boundary_min_size_m
        self.risk_adaptive = risk_adaptive
        self._hazard_lut = np.zeros(256, dtype=bool)
        for h in C.HAZARD_IDS:
            self._hazard_lut[h] = True
        # Resolution schedule sampled once; per-point targets come from an interp.
        self._lut_r = np.linspace(0.0, C.ROOT_SIZE_M, 1024)
        self._lut_t = np.array([C.target_cell_size(r) for r in self._lut_r])

    # ------------------------------------------------------------------ build --
    def build(self, xyz: np.ndarray, cls: np.ndarray, conf: np.ndarray) -> Grid:
        t0 = time.perf_counter()
        half = C.ROOT_SIZE_M / 2.0
        n_grid = 2 ** C.MAX_DEPTH

        keep = (np.abs(xyz[:, 0]) < half) & (np.abs(xyz[:, 1]) < half)
        xyz, cls, conf = xyz[keep], cls[keep], conf[keep]
        if xyz.shape[0] == 0:
            return self._finish(None, 0, t0)

        ix = np.clip(((xyz[:, 0] + half) / C.MIN_CELL_M).astype(np.int64), 0, n_grid - 1)
        iy = np.clip(((xyz[:, 1] + half) / C.MIN_CELL_M).astype(np.int64), 0, n_grid - 1)
        code = _morton(ix, iy)
        order = np.argsort(code, kind="stable")

        code = code[order]
        zs = xyz[order, 2].astype(np.float64)
        cs = np.clip(cls[order], 0, 255).astype(np.int64)
        cf = conf[order].astype(np.float64)
        rr = np.hypot(xyz[order, 0], xyz[order, 1])

        req, priority = self._required_level(rr, cs, cf)
        starts, ends, levels, drivers = self._extract_leaves(code, req, priority)
        grid = self._aggregate(starts, ends, levels, drivers, zs, cs, cf, code)
        return self._finish(grid, int(xyz.shape[0]), t0)

    # --------------------------------------------------- per-point resolution --
    def _required_level(self, rng_m, cls, conf):
        """The level each point needs, and why.

        Resolution is decided per point rather than per node: a point carries its own
        range, its own class and its own confidence, so the rule that refines a cell is
        attributable to a specific measurement rather than to a cell average.
        """
        target = np.interp(rng_m, self._lut_r, self._lut_t)
        priority = np.zeros(rng_m.shape, dtype=np.int64)     # 0 range

        if self.risk_adaptive:
            hazard = self._hazard_lut[cls]
            low = conf < self.conf_threshold
            # Hazard outranks low confidence when both apply: a known hazard is a harder
            # claim than an uncertain reading, and the operator should see it as such.
            t_haz = target / self.hazard_factor
            t_low = target / self.lowconf_factor
            target = np.where(low, np.minimum(target, t_low), target)
            priority = np.where(low, 2, priority)
            target = np.where(hazard, np.minimum(target, t_haz), target)
            priority = np.where(hazard, 3, priority)

        target = np.maximum(target, C.MIN_CELL_M)
        level = np.ceil(np.log2(C.ROOT_SIZE_M / target) - 1e-9).astype(np.int64)
        np.clip(level, 0, C.MAX_DEPTH, out=level)

        if self.risk_adaptive:
            # A point sitting on a class transition gets one extra level, bounded, so a
            # pole is not absorbed into the wall behind it where the range rule says
            # coarse is good enough.
            edge = np.zeros(cls.shape, dtype=bool)
            edge[1:] |= cls[1:] != cls[:-1]
            edge[:-1] |= cls[1:] != cls[:-1]
            cap = int(np.ceil(np.log2(C.ROOT_SIZE_M / self.boundary_min_size_m) - 1e-9))
            cap = min(cap, C.MAX_DEPTH)
            bumped = np.minimum(level + 1, cap)
            take = edge & (bumped > level)
            priority = np.where(take & (priority == 0), 1, priority)   # 1 = class boundary
            level = np.where(take, bumped, level)

        return level, priority

    # --------------------------------------------------------- leaf extraction --
    def _extract_leaves(self, code, req, priority):
        """Top-down sweep. At each level, a node whose points all need this level or
        coarser becomes a leaf; the rest carry on. Whole nodes are removed together, so
        every surviving node stays a contiguous run of the Morton-sorted array — which is
        what makes the alignment guarantee hold."""
        n = code.size
        idx = np.arange(n, dtype=np.int64)
        # priority -> driver id, ordered so the more specific reason wins a tie
        drv_of = np.array([C.DRIVER_RANGE, C.DRIVER_BOUNDARY,
                           C.DRIVER_CONFIDENCE, C.DRIVER_HAZARD], dtype=np.int64)
        key = req * 8 + priority

        s_all, e_all, l_all, d_all = [], [], [], []
        for lvl in range(C.MAX_DEPTH + 1):
            if idx.size == 0:
                break
            shift = np.uint32(2 * (C.MAX_DEPTH - lvl))
            pref = code[idx] >> shift
            gstart = np.concatenate([[0], np.flatnonzero(pref[1:] != pref[:-1]) + 1])
            gend = np.concatenate([gstart[1:], [idx.size]])
            gkey = np.maximum.reduceat(key[idx], gstart)
            gmax = gkey >> 3
            done = gmax <= lvl
            if done.any():
                s_all.append(idx[gstart[done]])
                e_all.append(idx[gend[done] - 1] + 1)
                l_all.append(np.full(int(done.sum()), lvl, dtype=np.int64))
                d_all.append(drv_of[gkey[done] & 7])
            if done.all():
                break
            counts = gend - gstart
            idx = idx[~np.repeat(done, counts)]

        if not s_all:
            return None, None, None, None
        starts = np.concatenate(s_all)
        o = np.argsort(starts, kind="stable")
        return (starts[o], np.concatenate(e_all)[o],
                np.concatenate(l_all)[o], np.concatenate(d_all)[o])

    # -------------------------------------------------------------- aggregate --
    def _aggregate(self, st, ends, levels, drivers, zs, cs, cf, code):
        if st is None:
            return None
        counts = (ends - st).astype(np.int64)
        half = C.ROOT_SIZE_M / 2.0

        # Cell centre straight from the Morton code: de-interleave the prefix at this
        # cell's level. No separate coordinate bookkeeping to drift out of step.
        shifts = (2 * (C.MAX_DEPTH - levels)).astype(np.int64)
        pref = code[st] >> shifts
        ix, iy = _demorton(pref)
        sizes = C.ROOT_SIZE_M / np.power(2.0, levels)
        cx = -half + (ix + 0.5) * sizes
        cy = -half + (iy + 0.5) * sizes

        cum_z = np.concatenate([[0.0], np.cumsum(zs)])
        cum_cf = np.concatenate([[0.0], np.cumsum(cf)])
        z_mean = (cum_z[ends] - cum_z[st]) / counts
        conf_mean = (cum_cf[ends] - cum_cf[st]) / counts
        z_min = np.minimum.reduceat(zs, st)
        z_max = np.maximum.reduceat(zs, st)

        # Most leaves hold a single class; only mixed ones need a majority vote.
        changes = np.concatenate([[0], np.cumsum(cs[1:] != cs[:-1])])
        mixed = changes[ends - 1] > changes[st]
        cls_out = cs[st].astype(np.uint8)
        for i in np.flatnonzero(mixed):
            cls_out[i] = np.bincount(cs[st[i]:ends[i]]).argmax()

        return Grid(
            x=cx.astype(np.float32),
            y=cy.astype(np.float32),
            level=levels.astype(np.uint8),
            z=z_mean.astype(np.float32),
            z_min=z_min.astype(np.float32),
            z_max=z_max.astype(np.float32),
            cls=cls_out,
            conf=conf_mean.astype(np.float32),
            driver=drivers.astype(np.uint8),
            n_points=counts.astype(np.int32),
            code=code[st],
        )

    # ------------------------------------------------------------------ stats --
    def _finish(self, grid, n_points, t0):
        if grid is None:
            z = np.zeros(0, dtype=np.float32)
            grid = Grid(z, z, np.zeros(0, np.uint8), z, z, z,
                        np.zeros(0, np.uint8), z, np.zeros(0, np.uint8),
                        np.zeros(0, np.int32), np.zeros(0, np.int64))
        grid.stats = self._stats(grid, n_points, (time.perf_counter() - t0) * 1000.0)
        return grid

    def _stats(self, grid: Grid, n_points: int, build_ms: float) -> dict:
        n_cells = len(grid)
        sizes = grid.size if n_cells else np.zeros(0)

        # Fair comparison: identical bytes per cell on both sides. The only thing being
        # compared is how many cells each representation needs for the same coverage.
        span = 2.0 * C.MAX_RANGE_M
        dense_cells = int((span / C.MIN_CELL_M) ** 2)
        # The PS compares against a "uniform high-resolution 3D map", so the 3D voxel
        # baseline is included too: the same 5 cm grid over a 10 m vertical extent.
        dense3d_voxels = dense_cells * int(10.0 / C.MIN_CELL_M)
        adaptive_bytes = n_cells * C.BYTES_PER_CELL
        dense_bytes = dense_cells * C.BYTES_PER_CELL
        dense3d_bytes = dense3d_voxels * 1                 # 1 byte per occupancy voxel
        cloud_bytes = n_points * C.BYTES_PER_POINT

        by_level, by_driver, by_class = {}, {}, {}
        if n_cells:
            for lvl, cnt in zip(*np.unique(grid.level, return_counts=True)):
                by_level[int(lvl)] = int(cnt)
            for d, cnt in zip(*np.unique(grid.driver, return_counts=True)):
                by_driver[C.DRIVER_NAMES.get(int(d), str(d))] = int(cnt)
            for c, cnt in zip(*np.unique(grid.cls, return_counts=True)):
                by_class[C.CLASS_NAMES.get(int(c), str(c))] = int(cnt)

        hazard_mask = np.isin(grid.cls, list(C.HAZARD_IDS)) if n_cells else np.zeros(0, bool)
        far = (np.hypot(grid.x, grid.y) > C.FOVEA_RADIUS_M) if n_cells else np.zeros(0, bool)

        return {
            "n_points": n_points,
            "n_cells": n_cells,
            "grid_build_ms": round(build_ms, 3),
            "cell_size_min_m": round(float(sizes.min()), 4) if n_cells else None,
            "cell_size_max_m": round(float(sizes.max()), 4) if n_cells else None,
            "cells_by_level": by_level,
            "cells_by_driver": by_driver,
            "cells_by_class": by_class,
            "risk_driven_cells": sum(v for k, v in by_driver.items() if k != "range"),
            "hazard_cells": int(hazard_mask.sum()),
            "hazard_cells_beyond_fovea": int((hazard_mask & far).sum()),
            "degraded_cells": int((grid.conf < self.conf_threshold).sum()) if n_cells else 0,
            "elevation_range_m": [round(float(grid.z.min()), 2),
                                  round(float(grid.z.max()), 2)] if n_cells else None,
            "memory": {
                "adaptive_bytes": adaptive_bytes,
                "dense_uniform_bytes": dense_bytes,
                "dense_uniform_cells": dense_cells,
                "dense_3d_voxel_bytes": dense3d_bytes,
                "dense_3d_voxels": dense3d_voxels,
                "reduction_vs_dense_3d": round(dense3d_bytes / adaptive_bytes, 1)
                                         if adaptive_bytes else None,
                "raw_cloud_bytes": cloud_bytes,
                "reduction_vs_dense": round(dense_bytes / adaptive_bytes, 1)
                                      if adaptive_bytes else None,
                "reduction_vs_cloud": round(cloud_bytes / adaptive_bytes, 2)
                                      if adaptive_bytes else None,
                "bytes_per_cell": C.BYTES_PER_CELL,
            },
        }


def uniform_baseline_cells(cell_size_m: float = C.MIN_CELL_M,
                           range_m: float = C.MAX_RANGE_M) -> int:
    """The PS-anchored baseline: a dense uniform grid at `cell_size_m` out to `range_m`."""
    return int((2.0 * range_m / cell_size_m) ** 2)


def decimate(grid: Grid, max_level: int, preserve_risk: bool = True) -> Grid:
    """Coarsen the map for transmission, leaving risk where it is.

    The onboard autonomy stack consumes the full-resolution grid. A remote node on a
    jam-constrained link does not need 5 cm road surface — but it does need the pothole.
    So cells whose resolution was set by a risk rule (hazard, degraded sensing, class
    boundary) are transmitted untouched, and everything else is merged up to `max_level`.

    This is lossy by construction and says so: the returned grid records what it dropped.
    """
    n = len(grid)
    if n == 0 or max_level >= int(grid.level.max()):
        out = Grid(grid.x, grid.y, grid.level, grid.z, grid.z_min, grid.z_max, grid.cls,
                   grid.conf, grid.driver, grid.n_points, grid.code, dict(grid.stats))
        out.stats["decimation"] = {"max_level": max_level, "merged_cells": 0,
                                   "preserved_risk_cells": 0}
        return out

    keep = grid.level <= max_level
    if preserve_risk:
        keep |= grid.driver != C.DRIVER_RANGE
    merge = ~keep
    n_merged_in = int(merge.sum())

    shift = np.uint32(2 * (C.MAX_DEPTH - max_level))
    pref = (grid.code[merge].astype(np.uint64) >> shift).astype(np.int64)
    order = np.argsort(pref, kind="stable")
    pref = pref[order]
    idx = np.flatnonzero(merge)[order]

    starts = np.concatenate([[0], np.flatnonzero(pref[1:] != pref[:-1]) + 1])
    ends = np.concatenate([starts[1:], [pref.size]])

    w = grid.n_points[idx].astype(np.float64)
    cum_w = np.concatenate([[0.0], np.cumsum(w)])
    cum_z = np.concatenate([[0.0], np.cumsum(grid.z[idx] * w)])
    cum_c = np.concatenate([[0.0], np.cumsum(grid.conf[idx] * w)])
    tot_w = cum_w[ends] - cum_w[starts]

    ix, iy = _demorton(pref[starts])
    size = C.ROOT_SIZE_M / (2.0 ** max_level)
    half = C.ROOT_SIZE_M / 2.0

    # Dominant class by point count, so a merged cell is labelled by what is actually in
    # it. Vectorised over a compact class index: a merged frame has thousands of groups
    # and this runs every frame.
    uniq, compact = np.unique(grid.cls[idx], return_inverse=True)
    gidx = np.repeat(np.arange(starts.size), ends - starts)
    tally = np.bincount(gidx * uniq.size + compact, weights=w,
                        minlength=starts.size * uniq.size).reshape(starts.size, uniq.size)
    cls_m = uniq[tally.argmax(axis=1)].astype(np.uint8)

    merged = Grid(
        x=(-half + (ix + 0.5) * size).astype(np.float32),
        y=(-half + (iy + 0.5) * size).astype(np.float32),
        level=np.full(starts.size, max_level, dtype=np.uint8),
        z=((cum_z[ends] - cum_z[starts]) / tot_w).astype(np.float32),
        z_min=np.minimum.reduceat(grid.z_min[idx], starts).astype(np.float32),
        z_max=np.maximum.reduceat(grid.z_max[idx], starts).astype(np.float32),
        cls=cls_m,
        conf=((cum_c[ends] - cum_c[starts]) / tot_w).astype(np.float32),
        driver=np.full(starts.size, C.DRIVER_RANGE, dtype=np.uint8),
        n_points=tot_w.astype(np.int32),
        code=(pref[starts] << shift).astype(np.int64),
    )

    kept_idx = np.flatnonzero(keep)
    out = Grid(
        x=np.concatenate([grid.x[kept_idx], merged.x]),
        y=np.concatenate([grid.y[kept_idx], merged.y]),
        level=np.concatenate([grid.level[kept_idx], merged.level]),
        z=np.concatenate([grid.z[kept_idx], merged.z]),
        z_min=np.concatenate([grid.z_min[kept_idx], merged.z_min]),
        z_max=np.concatenate([grid.z_max[kept_idx], merged.z_max]),
        cls=np.concatenate([grid.cls[kept_idx], merged.cls]),
        conf=np.concatenate([grid.conf[kept_idx], merged.conf]),
        driver=np.concatenate([grid.driver[kept_idx], merged.driver]),
        n_points=np.concatenate([grid.n_points[kept_idx], merged.n_points]),
        code=np.concatenate([grid.code[kept_idx], merged.code]),
        stats=dict(grid.stats),
    )
    out.stats["n_cells"] = len(out)
    out.stats["decimation"] = {
        "max_level": max_level,
        "max_cell_m": round(size, 3),
        "cells_before": n,
        "cells_after": len(out),
        "merged_from": n_merged_in,
        "merged_into": int(starts.size),
        "preserved_risk_cells": int((grid.driver[kept_idx] != C.DRIVER_RANGE).sum()),
        "preserve_risk": preserve_risk,
    }
    mem = dict(out.stats["memory"])
    mem["adaptive_bytes"] = len(out) * C.BYTES_PER_CELL
    mem["reduction_vs_dense"] = round(mem["dense_uniform_bytes"] / mem["adaptive_bytes"], 1)
    mem["reduction_vs_cloud"] = round(mem["raw_cloud_bytes"] / mem["adaptive_bytes"], 2)
    mem["reduction_vs_dense_3d"] = round(mem["dense_3d_voxel_bytes"] / mem["adaptive_bytes"], 1)
    out.stats["memory"] = mem
    return out
