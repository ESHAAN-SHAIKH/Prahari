"""
playback.py
===========
Live Playback Engine & End-to-End Pipeline Orchestration for PRAHARI-Lite.

Executes the real-time autonomous perception pipeline:
    Frame Loader -> Perception Backend (GT or Model) -> Variable-Resolution RingGrid
                                                     -> Uniform Baseline Grid (for comparison)

Maintains real achieved FPS and stage latency telemetry in memory.
Runs asynchronously in a non-blocking background thread/task.
"""

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from grid_engine.ring_grid import RingGrid
from grid_engine.serialize import grid_to_dict
from grid_engine.uniform_grid import UniformGrid
from perception.loader import get_frame_count
from perception.service import classify_frame, get_active_backend, set_active_backend

logger = logging.getLogger("playback.engine")


class PlaybackEngine:
    """
    Asynchronous sequence playback worker and pipeline orchestrator.
    """

    def __init__(self, sequence_dir: Optional[Path] = None, target_fps: float = 10.0):
        self.sequence_dir = sequence_dir
        self.target_fps = float(target_fps)
        self.speed_multiplier: float = 1.0

        self.total_frames = get_frame_count(sequence_dir=self.sequence_dir)
        self.current_frame_idx: int = 0
        self.is_playing: bool = False

        self.ring_grid = RingGrid()
        self.uniform_grid = UniformGrid()

        # Threading / concurrency locks
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Cached latest frames & telemetry
        self.latest_adaptive_frame: Optional[Dict[str, Any]] = None
        self.latest_uniform_frame: Optional[Dict[str, Any]] = None

        self.telemetry: Dict[str, Any] = {
            "current_frame": 0,
            "total_frames": self.total_frames,
            "is_playing": False,
            "target_fps": self.target_fps,
            "achieved_fps": 0.0,
            "speed_multiplier": 1.0,
            "active_backend": get_active_backend(),
            "latency_ms": {
                "perception": 0.0,
                "grid_adaptive": 0.0,
                "grid_uniform": 0.0,
                "serialization": 0.0,
                "total_pipeline": 0.0,
            },
            "memory_stats": {
                "adaptive_cells": 0,
                "uniform_cells": 0,
                "reduction_pct": 0.0,
            },
        }

        # Initialize with frame 0 precomputed
        self._process_single_frame(0)

    def _process_single_frame(self, frame_idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute full perception -> grid engine pipeline on a frame.
        """
        t_start = time.perf_counter()

        # 1. Perception Step
        t_p0 = time.perf_counter()
        points, class_ids = classify_frame(frame_idx, sequence_dir=self.sequence_dir)
        dt_perception = (time.perf_counter() - t_p0) * 1000.0

        # 2. Adaptive Ring Grid Step
        t_g0 = time.perf_counter()
        adaptive_cells = self.ring_grid.project(points, class_ids, max_range=100.0)
        dt_grid_adaptive = (time.perf_counter() - t_g0) * 1000.0

        # 3. Uniform Grid Step (Baseline comparison)
        t_u0 = time.perf_counter()
        uniform_cells = self.uniform_grid.project(points, class_ids, max_range=100.0)
        dt_grid_uniform = (time.perf_counter() - t_u0) * 1000.0

        # 4. Serialization Step
        t_s0 = time.perf_counter()
        adaptive_payload = grid_to_dict(adaptive_cells, frame_idx=frame_idx)
        uniform_payload = grid_to_dict(uniform_cells, frame_idx=frame_idx)
        dt_serialization = (time.perf_counter() - t_s0) * 1000.0

        dt_total = (time.perf_counter() - t_start) * 1000.0

        # 5. Compute reduction
        n_adapt = len(adaptive_cells)
        n_unif = len(uniform_cells)
        reduct_pct = (1.0 - (n_adapt / max(1, n_unif))) * 100.0

        with self._lock:
            self.current_frame_idx = frame_idx
            self.latest_adaptive_frame = adaptive_payload
            self.latest_uniform_frame = uniform_payload

            self.telemetry["current_frame"] = frame_idx
            self.telemetry["active_backend"] = get_active_backend()
            self.telemetry["latency_ms"]["perception"] = round(dt_perception, 2)
            self.telemetry["latency_ms"]["grid_adaptive"] = round(dt_grid_adaptive, 2)
            self.telemetry["latency_ms"]["grid_uniform"] = round(dt_grid_uniform, 2)
            self.telemetry["latency_ms"]["serialization"] = round(dt_serialization, 2)
            self.telemetry["latency_ms"]["total_pipeline"] = round(dt_total, 2)
            self.telemetry["memory_stats"]["adaptive_cells"] = n_adapt
            self.telemetry["memory_stats"]["uniform_cells"] = n_unif
            self.telemetry["memory_stats"]["reduction_pct"] = round(reduct_pct, 2)

        return adaptive_payload, uniform_payload

    def _playback_worker(self) -> None:
        """Background thread loop stepping through frames at target FPS."""
        logger.info("Playback worker loop started.")
        last_time = time.perf_counter()

        while not self._stop_event.is_set():
            if not self.is_playing:
                time.sleep(0.05)
                continue

            t_loop_start = time.perf_counter()

            # Next frame index with clean modulo wrap-around at sequence end
            next_idx = (self.current_frame_idx + 1) % max(1, self.total_frames)
            self._process_single_frame(next_idx)

            # Compute achieved FPS
            now = time.perf_counter()
            dt_loop = now - last_time
            last_time = now
            achieved = 1.0 / dt_loop if dt_loop > 0 else 0.0

            with self._lock:
                self.telemetry["achieved_fps"] = round(achieved, 1)

            # Frame rate pacing
            target_interval = 1.0 / (max(0.1, self.target_fps * self.speed_multiplier))
            elapsed = time.perf_counter() - t_loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Playback worker loop stopped.")

    def start(self) -> None:
        """Start or resume live sequence playback."""
        with self._lock:
            self.is_playing = True
            self.telemetry["is_playing"] = True

        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._playback_worker, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Pause playback."""
        with self._lock:
            self.is_playing = False
            self.telemetry["is_playing"] = False

    def seek(self, frame_idx: int) -> Dict[str, Any]:
        """Seek directly to a specific frame."""
        safe_idx = max(0, min(frame_idx, self.total_frames - 1))
        adapt, _ = self._process_single_frame(safe_idx)
        return adapt

    def set_speed(self, speed: float) -> None:
        """Set playback speed multiplier (e.g. 0.5x, 1.0x, 2.0x)."""
        with self._lock:
            self.speed_multiplier = max(0.1, float(speed))
            self.telemetry["speed_multiplier"] = self.speed_multiplier

    def get_latest_frame(self) -> Dict[str, Any]:
        """Return latest computed adaptive grid payload."""
        with self._lock:
            if self.latest_adaptive_frame is None:
                self._process_single_frame(self.current_frame_idx)
            return self.latest_adaptive_frame  # type: ignore

    def get_latest_uniform_frame(self) -> Dict[str, Any]:
        """Return latest computed uniform baseline grid payload."""
        with self._lock:
            if self.latest_uniform_frame is None:
                self._process_single_frame(self.current_frame_idx)
            return self.latest_uniform_frame  # type: ignore

    def get_telemetry(self) -> Dict[str, Any]:
        """Return live latency, throughput, and memory stats."""
        with self._lock:
            return dict(self.telemetry)


# ---------------------------------------------------------------------------
# Global Playback Engine Singleton
# ---------------------------------------------------------------------------
engine = PlaybackEngine()
