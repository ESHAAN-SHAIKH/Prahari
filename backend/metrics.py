"""Reads measured evidence produced by the TRINETRA training notebook.

The contract is deliberately one-directional: this module reads what the notebook wrote and
reports it with its provenance attached. It never computes an accuracy figure of its own and
never substitutes a plausible default for a missing one.

When a number has not been measured, the dashboard is told exactly that, together with the
command that would produce it. An unmeasured metric and a measured one must not be able to
look alike on screen.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config as C

# What each block needs, and how to get it, when it is missing.
HOW_TO_PRODUCE = {
    "per_class": "Notebook Phase 4.1 — ps_category_table(seg_per_class, det_per_class)",
    "distance": "Notebook Phase 4.2 — run_distance_eval(frnet_scan_iterator(...), label=...)",
    "edge": "Notebook Phase 6.3 — bash jetson_benchmark.sh on the Orin Nano, then "
            "ingest_jetson_results()",
    "baselines": "Notebook Phase 4.3 — fill BASELINES[...]['checkpoint'], then rerun",
    "training": "Notebook Phase 2.4 — plot_training_curve(FINETUNE_RUN)",
}


class Evidence:
    def __init__(self, path: Path = C.EVIDENCE_PATH):
        self.path = Path(path)
        self.dir = self.path.parent
        self.export: dict = {}
        self.per_class_files: list[dict] = []
        self.distance_files: list[dict] = []
        self.reload()

    # ---------------------------------------------------------------- loading --
    def reload(self) -> None:
        self.export = self._read(self.path) or {}
        self.per_class_files = [d for d in
                                (self._read(p) for p in sorted(self.dir.glob("ps_category_table_*.json")))
                                if d and "SELFTEST" not in str(d.get("label", ""))]
        self.distance_files = [d for d in
                               (self._read(p) for p in sorted(self.dir.glob("distance_stratified_*.json")))
                               if d and "SELFTEST" not in str(d.get("label", ""))]

    @staticmethod
    def _read(p: Path):
        try:
            return json.loads(Path(p).read_text())
        except Exception:
            return None

    @property
    def available(self) -> bool:
        return bool(self.export)

    def _ledger(self) -> list[dict]:
        return self.export.get("ledger", []) or []

    def _find(self, predicate) -> list[dict]:
        return [e for e in self._ledger() if predicate(e)]

    # ----------------------------------------------------------------- blocks --
    def per_class(self) -> dict:
        """Accuracy for the four categories the PS names."""
        if not self.per_class_files:
            return self._absent("per_class",
                                "Per-class accuracy for walls, poles, pedestrians and vehicles")
        runs = []
        for f in self.per_class_files:
            runs.append({
                "label": f.get("label", "run"),
                "aggregation": f.get("aggregation"),
                "created": f.get("created"),
                "rows": f.get("rows", []),
            })
        return {"status": "measured", "runs": runs,
                "caveats": self.export.get("data_caveats", [])}

    def distance(self) -> dict:
        """Accuracy stratified by range band — the PS's 'across varying distances'."""
        if self.distance_files:
            runs = [{"label": f.get("label"), "n_scans": f.get("n_scans"),
                     "command": f.get("command"), "summary": f.get("summary", {})}
                    for f in self.distance_files]
            return {"status": "measured", "runs": runs,
                    "caveats": self.export.get("data_caveats", [])}

        entries = self._find(lambda e: e.get("tag") == "distance")
        if entries:
            runs: dict[str, dict] = {}
            for e in entries:
                label, _, band = e["metric"].rpartition("_mIoU_")
                runs.setdefault(label, {"label": label, "summary": {}, "command": e["command"]})
                runs[label]["summary"][band] = {"mIoU_dataset": e["value"]}
            return {"status": "measured", "runs": list(runs.values()),
                    "caveats": self.export.get("data_caveats", [])}
        return self._absent("distance", "Accuracy by range band (0-10 m, 10-30 m, 30-100 m)")

    def edge(self) -> dict:
        """On-device latency, FPS and memory from the Jetson board."""
        entries = self._find(lambda e: e.get("tag") == "edge")
        if not entries:
            return self._absent("edge",
                                f"Measured latency, FPS and peak memory on {self.board()}")
        models: dict[str, dict] = {}
        for e in entries:
            for suffix, key in (("_latency_ms", "latency_ms"), ("_fps", "fps"),
                                ("_peak_mem_MB", "peak_mem_mb")):
                if e["metric"].endswith(suffix):
                    name = e["metric"][:-len(suffix)]
                    models.setdefault(name, {"model": name})
                    models[name][key] = e["value"]
                    models[name]["hardware"] = e.get("hardware")
                    models[name]["command"] = e["command"]
        return {"status": "measured", "models": list(models.values()),
                "board": self.board()}

    def baselines(self) -> dict:
        b = self.export.get("baselines") or {}
        named = {k: v for k, v in b.items() if v.get("checkpoint")}
        if not named:
            return self._absent("baselines", "Named baselines with the checkpoint for each")
        return {"status": "measured", "baselines": named,
                "comparisons": self.export.get("comparisons", [])}

    def segmentation(self) -> dict:
        entries = self._find(lambda e: e.get("tag") == "segmentation")
        if not entries:
            return self._absent("training", "Segmentation mIoU, baseline and fine-tuned")
        return {"status": "measured", "entries": entries,
                "caveats": self.export.get("data_caveats", [])}

    def checklist(self) -> dict:
        checks = self.export.get("checks") or {}
        if not checks:
            return {"status": "absent", "done": 0, "total": 0, "phases": {}}
        phases: dict[str, dict] = {}
        for cid, c in checks.items():
            p = cid.split(".")[0]
            phases.setdefault(p, {"done": 0, "total": 0, "items": []})
            phases[p]["total"] += 1
            phases[p]["done"] += 1 if c.get("done") else 0
            phases[p]["items"].append({"id": cid, "text": c.get("text"),
                                       "done": bool(c.get("done"))})
        return {"status": "measured",
                "done": sum(1 for c in checks.values() if c.get("done")),
                "total": len(checks), "phases": phases}

    # ------------------------------------------------------------------ misc --
    def board(self) -> str:
        return self.export.get("edge_board") or C_BOARD_DEFAULT

    def caveats(self) -> list[str]:
        return self.export.get("data_caveats", [])

    def _absent(self, key: str, what: str) -> dict:
        return {"status": "not_measured", "what": what,
                "how": HOW_TO_PRODUCE.get(key, ""),
                "evidence_path": str(self.path)}

    def snapshot(self) -> dict:
        return {
            "available": self.available,
            "generated": self.export.get("generated"),
            "evidence_path": str(self.path),
            "board": self.board(),
            "caveats": self.caveats(),
            "per_class": self.per_class(),
            "distance": self.distance(),
            "edge": self.edge(),
            "baselines": self.baselines(),
            "segmentation": self.segmentation(),
            "checklist": self.checklist(),
        }


C_BOARD_DEFAULT = "Jetson Orin Nano 8GB"
