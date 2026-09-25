"""PRAHARI prototype server.

One pipeline, three stages, timed individually every frame:

    perception  ->  variable-resolution grid  ->  codec  ->  link

The dashboard consumes the codec output, so what is rendered is what was transmitted.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config as C
from . import codec
from .grid_engine import GridEngine, decimate, uniform_baseline_cells
from .metrics import Evidence
from .perception import build_perception


# --------------------------------------------------------------------- state ---
@dataclass
class StreamSettings:
    risk_adaptive: bool = True
    compress: bool = True
    hz: float = C.TARGET_HZ
    conf_threshold: float = C.CONF_THRESHOLD
    hazard_factor: float = C.HAZARD_REFINE_FACTOR
    # None = send the onboard map as built. An integer caps the transmitted level, which
    # is what a remote node on a constrained link receives.
    link_tier: int | None = None
    preserve_risk_on_link: bool = True


@dataclass
class RollingStats:
    window: int = 60
    frame_ms: deque = field(default_factory=lambda: deque(maxlen=60))
    perception_ms: deque = field(default_factory=lambda: deque(maxlen=60))
    grid_ms: deque = field(default_factory=lambda: deque(maxlen=60))
    codec_ms: deque = field(default_factory=lambda: deque(maxlen=60))
    wire_bytes: deque = field(default_factory=lambda: deque(maxlen=60))

    def push(self, frame, perc, grid, cod, wire):
        self.frame_ms.append(frame)
        self.perception_ms.append(perc)
        self.grid_ms.append(grid)
        self.codec_ms.append(cod)
        self.wire_bytes.append(wire)

    @staticmethod
    def _p(d, q):
        return round(float(np.percentile(list(d), q)), 2) if d else None

    def summary(self):
        mean = lambda d: round(float(np.mean(list(d))), 2) if d else None  # noqa: E731
        fps = (1000.0 / mean(self.frame_ms)) if self.frame_ms and mean(self.frame_ms) else None
        return {
            "fps": round(fps, 2) if fps else None,
            "frame_ms_mean": mean(self.frame_ms),
            "frame_ms_p95": self._p(self.frame_ms, 95),
            "perception_ms": mean(self.perception_ms),
            "grid_ms": mean(self.grid_ms),
            "codec_ms": mean(self.codec_ms),
            "wire_bytes_mean": int(np.mean(list(self.wire_bytes))) if self.wire_bytes else None,
            "samples": len(self.frame_ms),
            "history_ms": [round(v, 2) for v in self.frame_ms],
        }


class Pipeline:
    def __init__(self):
        self.perception, self.fallback_note = build_perception()
        self.engine = GridEngine()
        self.frame_id = 0
        self.stats = RollingStats()

    def step(self, s: StreamSettings) -> tuple[bytes, dict]:
        t_frame = time.perf_counter()

        p = self.perception.infer(self.frame_id)

        self.engine.risk_adaptive = s.risk_adaptive
        self.engine.conf_threshold = s.conf_threshold
        self.engine.hazard_factor = s.hazard_factor
        t0 = time.perf_counter()
        grid = self.engine.build(p["xyz"], p["cls"], p["conf"])
        grid_ms = (time.perf_counter() - t0) * 1000.0

        tx = grid
        decim_ms = 0.0
        if s.link_tier is not None:
            t0 = time.perf_counter()
            tx = decimate(grid, s.link_tier, preserve_risk=s.preserve_risk_on_link)
            decim_ms = (time.perf_counter() - t0) * 1000.0

        frame_ms = (time.perf_counter() - t_frame) * 1000.0
        rolling = self.stats.summary()
        hz_effective = rolling["fps"] or s.hz

        tail = {
            "stats": tx.stats,
            "onboard": {"n_cells": len(grid),
                        "memory": grid.stats["memory"],
                        "cells_by_level": grid.stats["cells_by_level"],
                        "cells_by_driver": grid.stats["cells_by_driver"]},
            "detections": p.get("detections", []),
            "source": p["source"],
            "source_label": p.get("label"),
            "notes": p.get("notes"),
            "dust_center_deg": p.get("dust_center_deg"),
            "scan": p.get("scan"),
            "timing": {
                "perception_ms": p["latency_ms"],
                "grid_ms": round(grid_ms, 3),
                "decimate_ms": round(decim_ms, 3),
                "frame_ms": round(frame_ms, 3),
            },
            "rolling": rolling,
            "settings": {"risk_adaptive": s.risk_adaptive, "compress": s.compress,
                         "conf_threshold": s.conf_threshold,
                         "hazard_factor": s.hazard_factor,
                         "link_tier": s.link_tier,
                         "preserve_risk_on_link": s.preserve_risk_on_link},
        }

        t0 = time.perf_counter()
        wire, info = codec.encode(tx, self.frame_id, tail, compress=s.compress)
        codec_ms = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - t_frame) * 1000.0
        self.stats.push(total_ms, p["latency_ms"], grid_ms, codec_ms, info["wire_bytes"])
        self.frame_id += 1

        meta = {**info, **codec.link_budget(info["wire_bytes"], hz_effective),
                "frame_ms": round(total_ms, 3)}
        return wire, meta

    def compare(self) -> dict:
        """Run one frame three ways. This is the A/B that makes the novelty measurable
        rather than asserted."""
        p = self.perception.infer(self.frame_id)
        out = {}
        for name, risk in (("risk_adaptive", True), ("range_only", False)):
            eng = GridEngine(risk_adaptive=risk)
            g = eng.build(p["xyz"], p["cls"], p["conf"])
            wire, info = codec.encode(g, 0, {"stats": g.stats})
            hazard_mask = np.isin(g.cls, list(C.HAZARD_IDS))
            far_hazard = hazard_mask & (np.hypot(g.x, g.y) > C.FOVEA_RADIUS_M)
            out[name] = {
                "n_cells": len(g),
                "grid_ms": g.stats["grid_build_ms"],
                "wire_bytes": info["wire_bytes"],
                "memory": g.stats["memory"],
                "cells_by_driver": g.stats["cells_by_driver"],
                # The number that matters: resolution actually delivered on hazards
                # beyond the fovea, where range alone would have left them coarse.
                "hazard_cells_beyond_fovea": int(far_hazard.sum()),
                "median_hazard_cell_m": (round(float(np.median(g.size[far_hazard])), 3)
                                         if far_hazard.any() else None),
                "finest_hazard_cell_m": (round(float(g.size[far_hazard].min()), 3)
                                         if far_hazard.any() else None),
                "degraded_cells": g.stats["degraded_cells"],
            }
        full = GridEngine(risk_adaptive=True).build(p["xyz"], p["cls"], p["conf"])
        tiers = {}
        for lvl in (11, 10, 9, 8):
            for preserve in (True, False):
                d = decimate(full, lvl, preserve_risk=preserve)
                _, inf = codec.encode(d, 0, {})
                haz = np.isin(d.cls, list(C.HAZARD_IDS)) & (np.hypot(d.x, d.y) > C.FOVEA_RADIUS_M)
                tiers[f"level{lvl}_{'risk_preserved' if preserve else 'plain'}"] = {
                    "max_cell_m": round(C.ROOT_SIZE_M / 2 ** lvl, 3),
                    "n_cells": len(d),
                    "wire_bytes": inf["wire_bytes"],
                    "mbps_at_10hz": codec.link_budget(inf["wire_bytes"], 10.0)["mbps"],
                    "median_hazard_cell_m": (round(float(np.median(d.size[haz])), 3)
                                             if haz.any() else None),
                }
        out["transmit_tiers"] = tiers

        dense_cells = uniform_baseline_cells()
        out["dense_uniform_5cm"] = {
            "n_cells": dense_cells,
            "bytes": dense_cells * C.BYTES_PER_CELL,
            "note": "PS-anchored baseline: uniform 5 cm cells to 100 m, same bytes per cell.",
        }
        out["raw_point_cloud"] = {
            "n_points": int(p["xyz"].shape[0]),
            "bytes": int(p["xyz"].shape[0]) * C.BYTES_PER_POINT,
            "note": "x, y, z, intensity as float32.",
        }
        return out


# ----------------------------------------------------------------------- app ---
app = FastAPI(title="PRAHARI", version="0.1.0",
              description="Adaptive variable-resolution 2.5D LiDAR mapping — prototype")
pipeline = Pipeline()
evidence = Evidence()


@app.get("/api/system")
def system():
    return {
        "name": "PRAHARI",
        "subtitle": "Adaptive variable-resolution 2.5D LiDAR mapping",
        "problem_statement": "DRDO 26053",
        "geometry": {
            "root_size_m": C.ROOT_SIZE_M,
            "max_depth": C.MAX_DEPTH,
            "min_cell_m": C.MIN_CELL_M,
            "max_range_m": C.MAX_RANGE_M,
            "fovea_radius_m": C.FOVEA_RADIUS_M,
            "coarse_size_m": C.COARSE_SIZE_M,
        },
        "level_table": C.level_table(),
        "classes": [{"id": k, "name": v[0], "group": v[1], "hazard": v[2]}
                    for k, v in C.CLASSES.items()],
        "ps_categories": C.PS_CATEGORIES,
        "drivers": C.DRIVER_NAMES,
        "risk_rules": {
            "hazard_refine_factor": C.HAZARD_REFINE_FACTOR,
            "lowconf_refine_factor": C.LOWCONF_REFINE_FACTOR,
            "conf_threshold": C.CONF_THRESHOLD,
            "boundary_min_size_m": C.BOUNDARY_MIN_SIZE_M,
        },
        "perception": {
            "source": pipeline.perception.source,
            "label": getattr(pipeline.perception, "label", None),
            "notes": getattr(pipeline.perception, "notes", None),
            "fallback_note": pipeline.fallback_note,
        },
        "target_hz": C.TARGET_HZ,
        "evidence_available": evidence.available,
    }


@app.get("/api/metrics")
def metrics(reload: bool = False):
    if reload:
        evidence.reload()
    return evidence.snapshot()


@app.get("/api/compare")
def compare():
    return pipeline.compare()


@app.get("/api/frame")
def frame(compress: bool = True, risk_adaptive: bool = True, link_tier: int | None = None):
    """One frame over plain HTTP, in the same wire format as the stream."""
    wire, meta = pipeline.step(StreamSettings(risk_adaptive=risk_adaptive,
                                              compress=compress, link_tier=link_tier))
    return Response(content=wire, media_type="application/octet-stream",
                    headers={"X-Prahari-Meta": json.dumps(meta)})


@app.get("/api/health")
def health():
    return {"ok": True, "frames_served": pipeline.frame_id,
            "source": pipeline.perception.source}


@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    s = StreamSettings()
    loop = asyncio.get_running_loop()
    paused = False

    async def read_commands():
        nonlocal paused
        try:
            while True:
                msg = await ws.receive_text()
                cmd = json.loads(msg)
                if cmd.get("cmd") == "set":
                    for k in ("risk_adaptive", "compress", "conf_threshold",
                              "hazard_factor", "hz", "link_tier",
                              "preserve_risk_on_link"):
                        if k in cmd:
                            setattr(s, k, cmd[k])
                elif cmd.get("cmd") == "pause":
                    paused = True
                elif cmd.get("cmd") == "resume":
                    paused = False
        except (WebSocketDisconnect, RuntimeError):
            pass

    reader = asyncio.create_task(read_commands())
    try:
        while True:
            t0 = loop.time()
            if not paused:
                wire, meta = await loop.run_in_executor(None, pipeline.step, s)
                await ws.send_bytes(wire)
            delay = max(0.0, (1.0 / max(s.hz, 0.5)) - (loop.time() - t0))
            await asyncio.sleep(delay)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        reader.cancel()


# ------------------------------------------------------------------ frontend ---
if C.FRONTEND_DIR.exists():
    app.mount("/css", StaticFiles(directory=str(C.FRONTEND_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(C.FRONTEND_DIR / "js")), name="js")

    @app.get("/")
    def index():
        return FileResponse(str(C.FRONTEND_DIR / "index.html"))
else:  # pragma: no cover
    @app.get("/")
    def index():
        return JSONResponse({"error": "frontend/ not found next to backend/"}, status_code=500)
