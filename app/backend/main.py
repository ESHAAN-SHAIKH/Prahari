"""
main.py
=======
FastAPI Backend Application for PRAHARI-Lite.

Exposes REST endpoints for:
    - Frame Streaming: GET /frame, GET /frame/uniform, GET /frame/{idx}
    - Playback Controls: POST /playback/start, POST /playback/stop, POST /playback/speed, POST /playback/seek
    - Perception Switching: POST /perception/backend
    - Health & Telemetry: GET /health, GET /status, GET /metrics
"""

from pathlib import Path
import sys
from typing import Literal, Optional

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.backend.playback import engine
from perception.loader import get_frame_count
from perception.service import get_active_backend, set_active_backend

# ---------------------------------------------------------------------------
# FastAPI Application Instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PRAHARI-Lite Autonomous Defense Perception API",
    description="Adaptive Variable-Resolution 2.5D Grid Perception & Real-Time Telemetry Service",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------------------------
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8501",  # Streamlit fallback
    "http://127.0.0.1:8501",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health & Status Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health_check():
    """Service health probe endpoint."""
    return {
        "status": "ok",
        "service": "prahari-lite-backend",
        "version": "0.1.0",
    }


@app.get("/status", tags=["Status"])
async def system_status():
    """Return runtime system configuration, sequence metadata, and playback status."""
    total_frames = get_frame_count()
    active_backend = get_active_backend()

    return {
        "status": "online",
        "total_frames": total_frames,
        "active_backend": active_backend,
        "is_playing": engine.is_playing,
        "current_frame": engine.current_frame_idx,
        "speed_multiplier": engine.speed_multiplier,
        "ring_tiers": [
            {"tier": 0, "max_range_m": 10.0, "cell_size_m": 0.05},
            {"tier": 1, "max_range_m": 30.0, "cell_size_m": 0.15},
            {"tier": 2, "max_range_m": 100.0, "cell_size_m": 0.50},
        ],
    }


# ---------------------------------------------------------------------------
# Frame Data Endpoints
# ---------------------------------------------------------------------------

@app.get("/frame", tags=["Frames"])
async def get_current_adaptive_frame():
    """
    Get the latest computed variable-resolution adaptive 2.5D grid frame.
    """
    return engine.get_latest_frame()


@app.get("/frame/uniform", tags=["Frames"])
async def get_current_uniform_frame():
    """
    Get the latest frame's uniform high-resolution baseline grid (for comparison view).
    """
    return engine.get_latest_uniform_frame()


@app.get("/frame/{frame_idx}", tags=["Frames"])
async def get_frame_by_index(frame_idx: int):
    """
    Compute and retrieve the adaptive grid for a specific frame index.
    """
    if frame_idx < 0 or frame_idx >= engine.total_frames:
        raise HTTPException(
            status_code=404,
            detail=f"Frame index {frame_idx} out of range [0, {engine.total_frames - 1}]",
        )
    return engine.seek(frame_idx)


# ---------------------------------------------------------------------------
# Playback Control Endpoints
# ---------------------------------------------------------------------------

@app.post("/playback/start", tags=["Playback"])
async def start_playback():
    """Start or resume background sequence playback."""
    engine.start()
    return {
        "status": "playing",
        "current_frame": engine.current_frame_idx,
        "speed": engine.speed_multiplier,
    }


@app.post("/playback/stop", tags=["Playback"])
async def stop_playback():
    """Pause background sequence playback."""
    engine.stop()
    return {
        "status": "stopped",
        "current_frame": engine.current_frame_idx,
    }


@app.post("/playback/speed", tags=["Playback"])
async def set_playback_speed(x: float = Query(1.0, ge=0.1, le=10.0, description="Speed multiplier")):
    """Set playback speed multiplier (e.g. 0.5x, 1.0x, 2.0x)."""
    engine.set_speed(x)
    return {
        "status": "ok",
        "speed_multiplier": engine.speed_multiplier,
    }


@app.post("/playback/seek", tags=["Playback"])
async def seek_playback(frame: int = Query(0, ge=0, description="Target frame index")):
    """Jump playback directly to a specific frame."""
    if frame < 0 or frame >= engine.total_frames:
        raise HTTPException(
            status_code=400,
            detail=f"Target frame {frame} outside range [0, {engine.total_frames - 1}]",
        )
    engine.seek(frame)
    return {
        "status": "ok",
        "current_frame": engine.current_frame_idx,
    }


# ---------------------------------------------------------------------------
# Perception Backend Switch
# ---------------------------------------------------------------------------

@app.post("/perception/backend", tags=["Perception"])
async def switch_perception_backend(name: Literal["groundtruth", "model"] = Query(..., description="Backend name")):
    """Dynamically switch between groundtruth and model perception backends."""
    set_active_backend(name)
    # Re-evaluate current frame with new backend
    engine.seek(engine.current_frame_idx)
    return {
        "status": "ok",
        "active_backend": get_active_backend(),
    }


# ---------------------------------------------------------------------------
# Metrics Telemetry Endpoint
# ---------------------------------------------------------------------------

@app.get("/metrics", tags=["Metrics"])
async def get_pipeline_metrics():
    """Retrieve real-time latency, throughput, and memory telemetry."""
    return engine.get_telemetry()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=True)
