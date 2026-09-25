"""Synthetic LiDAR sensor model.

This exists so the prototype demonstrates the full pipeline with no checkpoint and no
80 GB dataset attached. It casts rays from a 64-channel spinning sensor into an analytic
scene (ground, walls, poles, vehicles, pedestrians, potholes) and returns labelled points
with per-point confidence, including a rotating dust wedge that degrades sensing in one
azimuth sector.

Everything produced here is labelled `source="synthetic"` all the way to the dashboard.
It is a sensor simulator, not a result.
"""
from __future__ import annotations

import numpy as np

from . import config as C


class Cylinder:
    __slots__ = ("x", "y", "r", "h", "cls", "vx", "vy")

    def __init__(self, x, y, r, h, cls, vx=0.0, vy=0.0):
        self.x, self.y, self.r, self.h, self.cls = x, y, r, h, cls
        self.vx, self.vy = vx, vy

    def at(self, t):
        return self.x + self.vx * t, self.y + self.vy * t


class Segment:
    __slots__ = ("x1", "y1", "x2", "y2", "h", "cls")

    def __init__(self, x1, y1, x2, y2, h, cls):
        self.x1, self.y1, self.x2, self.y2, self.h, self.cls = x1, y1, x2, y2, h, cls


class Depression:
    """A pothole/ditch: a circular patch of ground that is lower and differently classed."""
    __slots__ = ("x", "y", "r", "depth", "cls")

    def __init__(self, x, y, r, depth, cls):
        self.x, self.y, self.r, self.depth, self.cls = x, y, r, depth, cls


def default_scene() -> dict:
    """A rural-road scene with the hazards the PS cares about, placed deliberately.

    The pothole at 35 m is the demonstration case: far outside the 10 m fovea, so range
    alone would leave it coarse. It is what the risk-adaptive rule exists to catch.
    """
    walls = [
        Segment(-8.0, 6.0, 55.0, 7.5, 3.2, 4),        # compound wall, right side
        Segment(-30.0, -7.0, 20.0, -7.5, 2.4, 4),     # wall, left side
        Segment(58.0, -20.0, 62.0, 30.0, 4.0, 4),     # building face across the end
        Segment(20.0, -8.0, 46.0, -9.0, 1.0, 6),      # low barrier
    ]
    poles = [Cylinder(x, y, 0.16, 6.0, 5)
             for x, y in [(12, 6.2), (28, 6.6), (44, 7.0), (18, -7.1), (36, -7.3)]]
    poles += [Cylinder(52.0, 5.0, 0.12, 2.6, 5)]      # sign post

    vehicles = [
        Cylinder(22.0, 2.2, 1.05, 1.6, 8, vx=-3.2),   # oncoming truck
        Cylinder(-14.0, -2.4, 0.95, 1.5, 8, vx=1.8),  # overtaking car
        Cylinder(41.0, 1.8, 0.90, 1.5, 8, vx=-1.1),
    ]
    people = [
        Cylinder(9.5, -4.2, 0.32, 1.72, 7, vy=0.55),
        Cylinder(31.0, 4.6, 0.30, 1.66, 7, vx=-0.4, vy=-0.3),
        Cylinder(17.0, -5.6, 0.34, 1.78, 7, vy=0.2),
    ]
    riders = [Cylinder(-6.0, 3.0, 0.45, 1.6, 9, vx=4.0)]

    # Hazard placement is constrained by physics, not convenience: a 64-beam sensor's
    # ground rings are ~5 m apart at 37 m and ~10 m apart at 48 m, so a 1 m pothole out
    # there is simply not sampled. Hazards are sized and placed to land on rings — which
    # is also what broken pavement on a rural road actually looks like: patches, not pits.
    hazards = [
        Depression(36.9, 0.9, 2.60, 0.30, 40),        # broken pavement at 37 m: the case
        Depression(12.3, -1.8, 0.95, 0.22, 40),       # one inside the fovea, for contrast
        Depression(44.0, -5.0, 3.00, 0.50, 41),       # ditch along the verge
        Depression(25.0, -3.2, 2.00, 0.18, 43),       # water crossing
    ]
    rocks = [Cylinder(46.0, -2.0, 0.42, 0.55, 42), Cylinder(27.5, 5.2, 0.35, 0.45, 42),
             Cylinder(54.0, 4.2, 1.40, 0.60, 44)]     # snow drift, read as a low obstacle

    return {
        "segments": walls,
        "cylinders": poles + vehicles + people + riders + rocks,
        "depressions": hazards,
        "road_half_width": 5.0,
    }


class SyntheticLidar:
    """Casts a full sweep and returns (xyz, labels, confidence)."""

    def __init__(self, scene=None, seed: int = 7):
        self.scene = scene or default_scene()
        self.rng = np.random.default_rng(seed)
        az = np.linspace(-np.pi, np.pi, C.LIDAR_AZIMUTH_BINS, endpoint=False)
        el = np.radians(np.linspace(C.LIDAR_FOV_DOWN_DEG, C.LIDAR_FOV_UP_DEG,
                                    C.LIDAR_CHANNELS))
        self.az = np.repeat(az[None, :], C.LIDAR_CHANNELS, axis=0).ravel()
        self.el = np.repeat(el[:, None], C.LIDAR_AZIMUTH_BINS, axis=1).ravel()
        self.ca, self.sa = np.cos(self.az), np.sin(self.az)
        self.ce, self.se = np.cos(self.el), np.sin(self.el)
        self.n_rays = self.az.size

    # -- ground -------------------------------------------------------------
    def _ground(self):
        """Horizontal distance to the ground plane for each downward ray."""
        s = np.full(self.n_rays, np.inf, dtype=np.float64)
        down = self.se < -1e-4
        s[down] = (C.SENSOR_HEIGHT_M / -self.se[down]) * self.ce[down]
        return s

    # -- primitives ---------------------------------------------------------
    def _cylinders(self, t):
        best_s = np.full(self.n_rays, np.inf)
        best_c = np.zeros(self.n_rays, dtype=np.int32)
        for cy in self.scene["cylinders"]:
            cx, cyy = cy.at(t)
            b = cx * self.ca + cyy * self.sa
            c = cx * cx + cyy * cyy - cy.r * cy.r
            disc = b * b - c
            hit = disc > 0
            if not hit.any():
                continue
            s = np.where(hit, b - np.sqrt(np.maximum(disc, 0)), np.inf)
            s = np.where(s > 0.3, s, np.inf)
            z = s * np.tan(self.el)                      # height at the hit, sensor frame
            in_h = (z > -C.SENSOR_HEIGHT_M) & (z < -C.SENSOR_HEIGHT_M + cy.h)
            s = np.where(in_h, s, np.inf)
            closer = s < best_s
            best_s = np.where(closer, s, best_s)
            best_c = np.where(closer, cy.cls, best_c)
        return best_s, best_c

    def _segments(self):
        best_s = np.full(self.n_rays, np.inf)
        best_c = np.zeros(self.n_rays, dtype=np.int32)
        for sg in self.scene["segments"]:
            ex, ey = sg.x2 - sg.x1, sg.y2 - sg.y1
            denom = self.ca * ey - self.sa * ex
            ok = np.abs(denom) > 1e-9
            u = np.where(ok, (self.ca * (-sg.y1) - self.sa * (-sg.x1)) / np.where(ok, denom, 1), -1)
            s = np.where(ok & (u >= 0) & (u <= 1),
                         np.where(np.abs(self.ca) > np.abs(self.sa),
                                  (sg.x1 + u * ex) / np.where(np.abs(self.ca) > 1e-9, self.ca, 1e-9),
                                  (sg.y1 + u * ey) / np.where(np.abs(self.sa) > 1e-9, self.sa, 1e-9)),
                         np.inf)
            s = np.where(s > 0.3, s, np.inf)
            z = s * np.tan(self.el)
            in_h = (z > -C.SENSOR_HEIGHT_M) & (z < -C.SENSOR_HEIGHT_M + sg.h)
            s = np.where(in_h, s, np.inf)
            closer = s < best_s
            best_s = np.where(closer, s, best_s)
            best_c = np.where(closer, sg.cls, best_c)
        return best_s, best_c

    # -- sweep --------------------------------------------------------------
    def sweep(self, t: float, dust_center_deg: float | None = None,
              dust_width_deg: float = 44.0):
        gs = self._ground()
        cs, cc = self._cylinders(t)
        ss, sc = self._segments()

        s = gs.copy()
        cls = np.ones(self.n_rays, dtype=np.int32)       # ground defaults to drivable
        for cand_s, cand_c in ((cs, cc), (ss, sc)):
            closer = cand_s < s
            s = np.where(closer, cand_s, s)
            cls = np.where(closer, cand_c, cls)

        valid = np.isfinite(s) & (s > 0.5) & (s < C.MAX_RANGE_M)
        s = s[valid]
        cls = cls[valid].astype(np.int32)
        ca, sa = self.ca[valid], self.sa[valid]
        el = self.el[valid]

        x = s * ca
        y = s * sa
        z = s * np.tan(el)

        ground = cls == 1
        # Terrain: the verge is rough, the road corridor is drivable, with gentle undulation.
        hw = self.scene["road_half_width"]
        off_road = ground & (np.abs(y) > hw)
        cls[off_road] = 2
        verge = ground & (np.abs(y) > hw * 1.6)
        cls[verge] = 3
        undulation = 0.09 * np.sin(x * 0.11) + 0.05 * np.cos(y * 0.31)
        z = np.where(ground, z + undulation, z)

        # Depressions: lower the ground and relabel it as the hazard class.
        for d in self.scene["depressions"]:
            m = ground & (((x - d.x) ** 2 + (y - d.y) ** 2) < d.r ** 2)
            if m.any():
                rr = np.sqrt((x[m] - d.x) ** 2 + (y[m] - d.y) ** 2) / d.r
                z[m] -= d.depth * (1.0 - rr ** 2)
                cls[m] = d.cls

        rng_m = np.sqrt(x * x + y * y + z * z)
        conf = self._confidence(rng_m, cls, np.degrees(np.arctan2(y, x)),
                               dust_center_deg, dust_width_deg)

        # Dust also destroys returns, not just confidence.
        if dust_center_deg is not None:
            keep = self.rng.random(rng_m.size) > (0.55 * (conf < 0.45))
            x, y, z, cls, conf, rng_m = (a[keep] for a in (x, y, z, cls, conf, rng_m))

        # Sensor noise, scaled with range the way a real unit behaves.
        noise = self.rng.normal(0.0, 0.004 + 0.0011 * rng_m, size=rng_m.shape)
        z = z + noise

        xyz = np.stack([x, y, z], axis=1).astype(np.float32)
        return xyz, cls.astype(np.int32), conf.astype(np.float32)

    def _confidence(self, rng_m, cls, az_deg, dust_center_deg, dust_width_deg):
        base = np.full(rng_m.shape, 0.94, dtype=np.float64)
        base[cls == 2] = 0.86
        base[cls == 3] = 0.78
        base[np.isin(cls, list(C.HAZARD_IDS))] = 0.72     # hazards are genuinely harder
        base[cls == 7] = 0.83                             # small, sparse returns
        conf = base * (1.0 - 0.35 * (rng_m / C.MAX_RANGE_M) ** 1.5)

        if dust_center_deg is not None:
            d = np.abs(((az_deg - dust_center_deg + 180.0) % 360.0) - 180.0)
            inside = d < (dust_width_deg / 2.0)
            falloff = np.clip(1.0 - d / (dust_width_deg / 2.0), 0, 1)
            conf = np.where(inside, conf * (1.0 - 0.72 * falloff), conf)

        conf += self.rng.normal(0, 0.02, size=conf.shape)
        return np.clip(conf, 0.02, 0.995)


def detections(scene: dict, t: float) -> list[dict]:
    """Object-level boxes, the detection half of the pipeline.

    With a real PillarNet checkpoint attached these come from the detector; here they are
    read straight off the scene, which is why they carry source='synthetic' upstream.
    """
    out = []
    for cy in scene["cylinders"]:
        if cy.cls not in (7, 8, 9):
            continue
        x, y = cy.at(t)
        r = float(np.hypot(x, y))
        if r > C.MAX_RANGE_M:
            continue
        out.append({
            "x": round(float(x), 2), "y": round(float(y), 2),
            "r": round(r, 2), "cls": int(cy.cls),
            "w": round(cy.r * 2, 2), "h": round(cy.h, 2),
            "speed": round(float(np.hypot(cy.vx, cy.vy)), 2),
            # Detector score falls off with range the way a real one does.
            "score": round(float(np.clip(0.95 - 0.006 * r, 0.25, 0.97)), 3),
        })
    return sorted(out, key=lambda d: d["r"])
