"""
main.py
=======
FastAPI Backend Application for PRAHARI-Lite.

Exposes REST and WebSocket endpoints for LiDAR sequence playback,
variable-resolution 2.5D grid streaming, perception backend switching,
and real-time performance telemetry.
"""

from pathlib import Path
import sys

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from perception.loader import get_frame_count
from perception.service import get_active_backend

# ---------------------------------------------------------------------------
# FastAPI Application Instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PRAHARI-Lite Autonomous Defense Perception API",
    description="Adaptive Variable-Resolution 2.5D Grid Perception & Real-Time Telemetry Service",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS Configuration (Localhost / Dashboard Origins)
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
# Core Service Endpoints
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
    """Return runtime system configuration and sequence metadata."""
    total_frames = get_frame_count()
    active_backend = get_active_backend()

    return {
        "status": "online",
        "total_frames": total_frames,
        "active_backend": active_backend,
        "ring_tiers": [
            {"tier": 0, "max_range_m": 10.0, "cell_size_m": 0.05},
            {"tier": 1, "max_range_m": 30.0, "cell_size_m": 0.15},
            {"tier": 2, "max_range_m": 100.0, "cell_size_m": 0.50},
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=True)
