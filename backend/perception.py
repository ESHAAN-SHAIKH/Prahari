"""Perception stage: point cloud in, per-point class + confidence out.

Two implementations behind one interface.

`CheckpointPerception` runs the FRNet segmentation checkpoint your training notebook
produces, over real scans, and pulls the frustum-head confidence out of the same forward
pass — the signal the risk-adaptive grid consumes. This is the path that runs in the
prototype once PRAHARI_CKPT, PRAHARI_CFG and PRAHARI_SCANS are set.

`SyntheticPerception` runs an analytic sensor model instead, so the system demonstrates
end to end with nothing attached. Every frame it produces is stamped `source="synthetic"`,
which the dashboard surfaces as a badge that cannot be turned off. The two are never mixed
and a synthetic frame is never presented as a measured one.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from . import config as C
from .scene import SyntheticLidar, default_scene, detections


class PerceptionResult(dict):
    """xyz, cls, conf, detections, source, latency_ms, notes."""


class SyntheticPerception:
    source = "synthetic"
    label = "Synthetic sensor model"

    def __init__(self, seed: int = 7):
        self.scene = default_scene()
        self.lidar = SyntheticLidar(self.scene, seed=seed)
        self.notes = ("Analytic sensor model. Geometry and labels are generated, not "
                      "inferred. Accuracy figures cannot come from this source.")

    def infer(self, frame_idx: int) -> PerceptionResult:
        t0 = time.perf_counter()
        t = (frame_idx % 240) / 10.0                  # objects loop over 24 s
        dust = (frame_idx * 0.9) % 360.0 - 180.0      # the degraded sector sweeps round
        xyz, cls, conf = self.lidar.sweep(t, dust_center_deg=dust)
        return PerceptionResult(
            xyz=xyz, cls=cls, conf=conf,
            detections=detections(self.scene, t),
            source=self.source, label=self.label, notes=self.notes,
            dust_center_deg=round(float(dust), 1),
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        )


class CheckpointPerception:
    source = "checkpoint"

    def __init__(self, checkpoint: str, config: str, scans_dir: str):
        self.checkpoint, self.config = checkpoint, config
        self.scans = sorted(Path(scans_dir).glob("*.bin"))
        if not self.scans:
            raise FileNotFoundError(f"No .bin scans under {scans_dir}")
        self.label = f"FRNet — {Path(checkpoint).name}"
        self.notes = (f"Live inference over {len(self.scans)} scans from {scans_dir}. "
                      f"Per-point confidence is the frustum head's max softmax.")
        self._model = None
        self._load()

    def _load(self):
        from mmdet3d.apis import init_model          # imported lazily, only on this path
        self._model = init_model(self.config, self.checkpoint, device="cuda:0")
        self._model.eval()

    def _frustum_confidence(self, feats):
        """The auxiliary-head signal FRNet already computes internally (Phase 5 of the
        training runbook). Exposed here rather than rebuilt."""
        import torch.nn.functional as F
        head = None
        for attr in ("frustum_head", "aux_head", "frustum_seg_head"):
            if hasattr(self._model, attr):
                head = getattr(self._model, attr)
                break
        if head is None:
            raise AttributeError(
                "No frustum/auxiliary head on this model. Do not substitute a constant — "
                "the risk-adaptive grid's confidence path depends on a real signal.")
        logits = head(feats)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        return F.softmax(logits, dim=1).max(dim=1).values

    def infer(self, frame_idx: int) -> PerceptionResult:
        import torch
        t0 = time.perf_counter()
        path = self.scans[frame_idx % len(self.scans)]
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

        from mmdet3d.apis import inference_segmentor
        with torch.no_grad():
            result = inference_segmentor(self._model, str(path))
        seg = result.pred_pts_seg.pts_semantic_mask.cpu().numpy()

        try:
            feats = self._model.extract_feat(
                torch.from_numpy(pts).cuda()[None], [result])
            fconf = self._frustum_confidence(feats)[0].detach().cpu().numpy()
            conf = _confidence_per_point(pts[:, :3], fconf)
        except Exception as exc:                       # surfaced, never silently defaulted
            conf = np.full(seg.shape, np.nan, dtype=np.float32)
            self.notes = (f"Frustum confidence unavailable ({exc}). Risk-adaptive "
                          f"subdivision is running on range and hazard only.")

        return PerceptionResult(
            xyz=pts[:, :3], cls=seg.astype(np.int32),
            conf=np.nan_to_num(conf, nan=1.0).astype(np.float32),
            detections=[], source=self.source, label=self.label, notes=self.notes,
            scan=str(path.name),
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        )


def _confidence_per_point(xyz: np.ndarray, frustum_conf: np.ndarray) -> np.ndarray:
    """Project each point into the range image and read its frustum's confidence."""
    H, W = frustum_conf.shape
    az = (np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) + 180.0) % 360.0
    w = np.clip((az / 360.0 * W).astype(int), 0, W - 1)
    r_xy = np.hypot(xyz[:, 0], xyz[:, 1])
    el = np.degrees(np.arctan2(xyz[:, 2], np.maximum(r_xy, 1e-6)))
    h = np.clip(((C.LIDAR_FOV_UP_DEG - el) /
                 (C.LIDAR_FOV_UP_DEG - C.LIDAR_FOV_DOWN_DEG) * (H - 1)).astype(int), 0, H - 1)
    return frustum_conf[h, w]


def build_perception():
    """Pick the real path when it is fully configured; otherwise the synthetic one.

    A partially configured checkpoint is an error, not a reason to quietly fall back —
    that is how a demo ends up showing synthetic data while everyone believes it is live.
    """
    ckpt, cfg = C.CHECKPOINT_PATH, C.CHECKPOINT_CONFIG
    scans = os.environ.get("PRAHARI_SCANS", "")
    if not any((ckpt, cfg, scans)):
        return SyntheticPerception(), None
    missing = [n for n, v in (("PRAHARI_CKPT", ckpt), ("PRAHARI_CFG", cfg),
                              ("PRAHARI_SCANS", scans)) if not v]
    if missing:
        return SyntheticPerception(), (
            f"Checkpoint mode is partly configured — {', '.join(missing)} not set. "
            f"Running the synthetic sensor model instead.")
    try:
        return CheckpointPerception(ckpt, cfg, scans), None
    except Exception as exc:
        return SyntheticPerception(), f"Checkpoint failed to load ({exc}). Running synthetic."
