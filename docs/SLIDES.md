# PRAHARI-Lite — Pitch & Presentation Deck

> **Problem Statement 26053:** Adaptive Spatial Representation for Autonomous Off-Road Defense Vehicles  
> **Team:** PRAHARI  
> **Duration:** 5–7 Minutes Pitch + Live Demo

---

## Slide 1 — Title & Executive Pitch

### **PRAHARI-Lite**
#### *Risk-Adaptive Variable-Resolution 2.5D Perception Grid for Autonomous Off-Road Mobility*

```
                 [ 32.4% MEMORY REDUCTION ]
        [ 100.0000% POINT CONSERVATION · ZERO DATA LOSS ]
         [ REAL-TIME MULTI-TIER VARIABLE-RESOLUTION GRID ]
```

- **Core Problem:** Autonomous defence vehicles operating in harsh, unstructured off-road terrain cannot afford uniform high-resolution grids ($16\text{M}$ cells / scan at $5\text{cm}$ over $100\text{m} = 732\text{ MB/frame}$).
- **Our Solution:** A concentric 3-tier variable-resolution grid ($5\text{cm}$ near-field, $15\text{cm}$ mid-range, $50\text{cm}$ far-field) with hazard priority majority voting and local height-variance risk subdivision.

---

## Slide 2 — Problem vs Architecture

### *Bridging Point-Cloud Segmentation and Real-Time Spatial Planning*

```
LiDAR Point Cloud (120k pts/scan)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Perception Layer (Hot-Swappable)                       │
│  ├─ Ground-Truth Fallback (21.9ms latency)             │
│  └─ Pretrained PointNet Neural Segmenter (PyTorch CPU) │
└────────────────────────────────────────────────────────┘
         │ (x, y, z, semantic_class_id)
         ▼
┌────────────────────────────────────────────────────────┐
│ Ring-Tier Grid Engine (Vectorized NumPy)               │
│  ├─ Tier 0: 0–10m @ 5cm (Micro-terrain / Pedestrians)  │
│  ├─ Tier 1: 10–30m @ 15cm (Obstacles / Vehicles)       │
│  ├─ Tier 2: 30–100m @ 50cm (Far-Horizon Awareness)     │
│  └─ ⭐ Risk-Adaptive Subdivider (Height-Var Potholes)   │
└────────────────────────────────────────────────────────┘
         │ Active cells (x, y, size, height, class_id)
         ▼
┌────────────────────────────────────────────────────────┐
│ Real-Time Tactical Dashboard (React + Canvas 2.5D)     │
│  ├─ Live Variable-Resolution Canvas Grid View          │
│  ├─ Side-by-Side Uniform Baseline Comparison View      │
│  └─ Live Stage-by-Stage Latency & FPS Telemetry        │
└────────────────────────────────────────────────────────┘
```

---

## Slide 3 — Key Technical Innovations

1. **Strict Spatial Disjoint Partitioning & Conservation Guarantee**
   - Boundary condition $r \le r_{\text{tier}}$ guarantees zero ambiguous assignments.
   - **Audit Result:** $32,394,253$ LiDAR points across $271$ sequence frames verified with **$100.0000\%$ conservation and zero boundary violations**.

2. **Hazard Priority Majority Voting**
   - High-consequence safety hazards (pedestrians, vehicles, obstacles) override drivable terrain majorities within shared grid cells, preventing dangerous hazard smoothing.

3. **⭐ Risk-Adaptive Local Subdivision Rule**
   - Upgrades system from purely distance-tiered to genuinely risk-adaptive: distant potholes, bumps, and curbs with high height variance ($\sigma_z^2 \ge 0.04$) are dynamically refined to $5\text{cm}$ sub-cells regardless of range.

---

## Slide 4 — Live Demonstration (2 Minutes)

*(Switching to live React Dashboard at `http://localhost:5173`)*

- **Action 1:** Stream real-time sequence playback with live adaptive grid rendering (tiny cells near origin, coarse cells far out).
- **Action 2:** Switch to **Compare View** showing the live Uniform $5\text{cm}$ baseline against the Adaptive grid with prominent **32.4% memory savings** overlay.
- **Action 3:** Review live telemetry panel: FPS, stage latencies (classify, project, serialize, total), and hot-swap perception backends on-the-fly.

---

## Slide 5 — Empirical Benchmark Results

*Measured across full 271-frame SemanticKITTI Sequence 04 on laptop CPU:*

| Metric | Uniform Baseline (5cm) | Adaptive Ring Grid (5/15/50cm) | Measured Advantage |
|:---|:---:|:---:|:---:|
| **Avg Active Cells / Frame** | 117,960 | **79,715** | **32.4% Reduction** |
| **Avg Memory Footprint** | 5.40 MB | **3.65 MB** | **1.75 MB Saved / scan** |
| **Dense Matrix Raster** | 16M cells (732 MB) | **79k cells (3.65 MB)** | **>99.5% Reduction** |
| **Point Conservation** | — | **100.0000%** | **32.39M / 32.39M pts** |
| **End-to-End Latency (p95)** | — | **1,085 ms (CPU)** | *Target <15ms on Jetson Orin* |

---

## Slide 6 — Scope Discipline: What We Built vs What's Next

### What We Built (Hackathon Scope)
- ✅ Fully vectorized 3-tier 2.5D Ring Grid Engine with hazard priority voting
- ✅ Risk-adaptive variance-triggered cell subdivision rule
- ✅ 100% verified point conservation audit across full 271-frame sequence
- ✅ Empirical memory comparison & benchmark pipeline
- ✅ FastAPI backend with live playback loop and stage latency instrumentation
- ✅ React canvas dashboard with live comparison view and tactical telemetry

### Production Deployment Roadmap (Next Milestones)
- 🚀 **Embedded CUDA / TensorRT Acceleration:** Move grid projection from NumPy CPU to CUDA kernels on NVIDIA Jetson Orin (<10ms target).
- 🚀 **Swarm Bandwidth Compression:** Delta-frame spatial compression for low-bandwidth mesh radio transmission between unmanned ground vehicles.
- 🚀 **Dynamic Local Costmap Integration:** Direct ROS 2 Nav2 costmap plugin export for autonomous off-road trajectory planning.
