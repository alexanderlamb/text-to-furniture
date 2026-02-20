"""Step 1 run data helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def list_runs(runs_dir: str) -> List[Dict[str, Any]]:
    root = Path(runs_dir)
    if not root.exists():
        return []

    rows: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir(), reverse=True):
        if not child.is_dir() or child.name in {"latest", ".uploads", "suites"}:
            continue

        manifest = read_json(child / "manifest.json")
        metrics = read_json(child / "metrics.json")
        strategy = str(manifest.get("strategy", "")).strip()
        status = str(metrics.get("status") or manifest.get("status") or "unknown")

        rows.append(
            {
                "run_id": str(manifest.get("run_id") or child.name),
                "run_dir": str(child),
                "design_name": str(manifest.get("design_name") or child.name),
                "strategy": strategy,
                "status": status,
                "created_utc": manifest.get("created_utc"),
                "manifest": manifest,
                "metrics": metrics,
                "is_step1": strategy == "openscad_step1_clean_slate",
            }
        )

    return rows


def artifact_files(run_dir: str) -> List[Path]:
    artifacts = Path(run_dir) / "artifacts"
    if not artifacts.exists():
        return []
    return [path for path in sorted(artifacts.rglob("*")) if path.is_file()]


__all__ = ["read_json", "list_runs", "artifact_files"]
