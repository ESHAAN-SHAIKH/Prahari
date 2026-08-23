# PRAHARI-Lite — Memory-Footprint Comparison Report

> **Metric Source:** Empirical evaluation across **271 frames** of LiDAR demo sequence.
> **Tested Invariant:** Adaptive grid cell count $\le$ Uniform grid cell count across 100% of frames.

---

## 🎯 Headline Memory Metrics

| Metric | Uniform Baseline (5 cm) | Adaptive Ring Grid (5/15/50 cm) | Savings / Reduction |
|:---|:---:|:---:|:---:|
| **Average Active Cells / Frame** | **117,960** | **79,715** | **32.4% reduction** |
| **Average Memory Footprint** | **5.40 MB** | **3.65 MB** | **1.75 MB saved / frame** |
| **Dense Grid Theoretical Upper Bound** | 732.4 MB (16M cells) | ~3.65 MB (active cells) | **>99.5% vs Dense Raster** |

### Reduction Statistics across Sequence
- **Mean Memory Reduction:** `32.42%`
- **Minimum Memory Reduction (Worst-case Frame):** `32.11%`
- **Maximum Memory Reduction (Best-case Frame):** `32.74%`

---

## 📊 Ring Tier Resolution Breakdown

| Tier | Range Radius | Cell Resolution | Purpose & Trade-off |
|:---|:---:|:---:|:---|
| **Tier 0 (Near-Field)** | $0.0\text{ m} - 10.0\text{ m}$ | **0.05 m (5 cm)** | Micro-terrain traversability & safety critical pedestrian / vehicle hazard detection. |
| **Tier 1 (Mid-Range)** | $10.0\text{ m} - 30.0\text{ m}$ | **0.15 m (15 cm)** | Medium-range static obstacle tracking & dynamic vehicle pathing. |
| **Tier 2 (Far-Field)** | $30.0\text{ m} - 100.0\text{ m}$ | **0.50 m (50 cm)** | Far-range horizon situational awareness with aggressive **100x cell count reduction**. |

---

## 🔍 Sample Frame-by-Frame Log

| Frame # | Valid LiDAR Points | Uniform 5cm Cells | Adaptive Grid Cells | Memory Saved (%) |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 119,515 | 117,985 | 79,536 | **32.59%** |
| 30 | 119,515 | 117,971 | 79,914 | **32.26%** |
| 60 | 119,523 | 117,977 | 79,789 | **32.37%** |
| 90 | 119,550 | 117,974 | 79,502 | **32.61%** |
| 120 | 119,557 | 117,924 | 79,546 | **32.54%** |
| 150 | 119,531 | 117,880 | 79,800 | **32.30%** |
| 180 | 119,567 | 117,999 | 79,546 | **32.59%** |
| 210 | 119,548 | 117,966 | 79,502 | **32.61%** |
| 240 | 119,483 | 117,909 | 79,895 | **32.24%** |
| 270 | 119,554 | 117,948 | 79,588 | **32.52%** |

---

## 🛡️ Correctness Guarantee
Every frame in the sequence satisfies the invariant:
$$\text{cell\_count}_{\text{adaptive}} \le \text{cell\_count}_{\text{uniform}}$$
Zero point loss and zero alignment boundary errors were observed across the entire audit.
