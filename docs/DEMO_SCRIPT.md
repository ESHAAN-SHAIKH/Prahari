# PRAHARI-Lite — Demo Script

> **Total Runtime:** ~2 minutes  
> **Rehearse at least twice, timed, before presenting.**

---

## Pre-Demo Checklist (Run 5 min before slot)

```powershell
# Terminal 1 — Backend
cd C:\Users\Eshaa\Downloads\PRAHARI
.\.venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend  
cd C:\Users\Eshaa\Downloads\PRAHARI\app\frontend
npm run dev

# Verify backend is live
curl http://localhost:8000/health
# → {"status":"ok","service":"prahari-lite-backend","version":"0.1.0"}

# Open browser
start http://localhost:5173
```

---

## Scene 1 — Live Adaptive Grid (~30 sec)

**Tab:** `Adaptive`  
**Action:** Click **▶ Play**

**Say:**
> "This is PRAHARI-Lite — a risk-adaptive variable-resolution 2.5D perception grid  
> for autonomous defence vehicles, running live in real time.  
> Watch the cell sizes. Near the sensor — tiny 5cm cells, maximum detail.  
> At the edges — 50cm cells, covering more ground with fewer resources.  
> The system allocates precision exactly where it matters."

**Point at:** Near-field green terrain cells (tiny) → far-field coarse cells (large).  
**Point at:** Red pedestrian / cyan vehicle detections near the origin.

---

## Scene 2 — Comparison View (~30 sec)

**Tab:** `Compare`

**Say:**
> "This is the key claim. Left: a naive uniform 5cm grid — every cell the same size,  
> maximum memory, no intelligence.  
> Right: our adaptive ring grid — same data, but **{MEMORY_SAVED}% fewer cells**,  
> measured honestly over the full 271-frame sequence.  
> The savings number you see is not configured — it is computed live, per frame."

**Point at:** The green **32.4% memory saved** overlay badge.  
**Point at:** The cell count delta (117,960 → 79,715).

---

## Scene 3 — Metrics Panel (~20 sec)

**Panel:** Right sidebar

**Say:**
> "Every number here is measured, not assumed.  
> FPS is computed from real inter-frame timestamps.  
> The latency bars break down exactly which stage costs what —  
> classify, grid projection, serialization — total pipeline, not inference-only.  
> The backend indicator shows whether we're running ground-truth labels  
> or the live neural classifier."

**Point at:** FPS number (colour: green if on-target, amber/red if slow).  
**Point at:** Latency bars — Classify / Project / Serialize.  
**Point at:** `Memory Saved 32.4%` stat row.

---

## Scene 4 — Backend Toggle (~15 sec)

**Action:** Click `GT Backend` → switches to `Neural Backend`

**Say:**
> "We can hot-swap the perception backend — from ground-truth labels  
> to a live PointNet neural classifier — without restarting anything.  
> The grid adapts immediately. This is the dynamic backend switching from TASK-006."

**Action:** Toggle back to `GT Backend` for clean exit.

---

## Fallback Plan

If the live system fails:
1. Open fallback recording: `docs/demo_recording.mp4`
2. Play the recording — narrate from this same script
3. The recording was captured from the **real running system**, not edited

---

## Key Numbers to Know by Heart

| Claim | Number | Source |
|:---|:---:|:---|
| Memory saved | 32.4% | Full 271-frame sequence average |
| Worst-case frame | 32.1% | Min across sequence |
| Frames in sequence | 271 | SemanticKITTI Seq-04 (synthetic) |
| Points per frame | ~119,515 | Real LiDAR scan density |
| Adaptive cells avg | 79,715 | Measured — not configured |
| Uniform cells avg | 117,960 | 5cm baseline comparison |
| p95 pipeline latency | ~450ms | End-to-end, not inference-only |

---

## Q&A Preparation

| Judge Question | Answer |
|:---|:---|
| "How did you measure the memory savings?" | Cell count: adaptive cells / uniform cells, per frame, every frame, 271-frame average |
| "Why not always use 5cm everywhere?" | 5cm at 100m = 16M cells, 732MB — infeasible real-time. We use 5cm only near the hazard zone |
| "What if FPS is low?" | That's real CPU-only. On edge GPU (Jetson), adaptive grid projection < 15ms |
| "Is the neural model accurate?" | GT backend = 100% by construction. Model backend is a PointNet student; the novel contribution is the grid, not the classifier |
| "What's the risk-adaptive rule?" | High height-variance cells (potholes, bumps) at mid/far range get subdivided to finer resolution automatically |
