"""Pure data helpers for the review UI. No Streamlit imports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file, returning {} if missing."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_runs(runs_dir: str) -> List[Dict[str, Any]]:
    """Discover run directories under *runs_dir*, newest first."""
    root = Path(runs_dir)
    if not root.exists():
        return []
    items: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir(), reverse=True):
        if not child.is_dir() or child.name in {"latest", ".uploads", "suites"}:
            continue
        payload: Dict[str, Any] = {
            "run_id": child.name,
            "status": "running",
            "design_name": child.name,
            "created_utc": None,
        }
        manifest = child / "manifest.json"
        if manifest.exists():
            manifest_data = read_json(manifest)
            if manifest_data:
                payload.update(manifest_data)
        payload["run_dir"] = str(child)
        items.append(payload)
    return items


def list_suite_runs(suites_dir: str) -> List[Dict[str, Any]]:
    """Discover suite run directories, newest first."""
    root = Path(suites_dir)
    if not root.exists():
        return []
    items: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir(), reverse=True):
        if not child.is_dir() or child.name == "latest":
            continue
        manifest = read_json(child / "manifest.json")
        results = read_json(child / "results.json")
        rows = results.get("rows", []) if isinstance(results.get("rows"), list) else []
        case_count = len(rows)
        executed = sum(
            1
            for row in rows
            if str(row.get("status", "")).lower() not in {"missing_input", "error"}
        )
        error_rows = sum(1 for row in rows if str(row.get("status", "")).lower() == "error")
        fail_rows = sum(1 for row in rows if str(row.get("status", "")).lower() == "fail")
        partial_rows = sum(
            1 for row in rows if str(row.get("status", "")).lower() == "partial"
        )
        success_rows = sum(
            1 for row in rows if str(row.get("status", "")).lower() == "success"
        )
        scores = [
            float(row.get("overall_score"))
            for row in rows
            if isinstance(row.get("overall_score"), (int, float))
        ]
        mean_score = float(sum(scores) / len(scores)) if scores else None
        items.append(
            {
                "suite_run_id": child.name,
                "suite_dir": str(child),
                "suite_name": manifest.get("suite_name", child.name),
                "created_utc": manifest.get("created_utc"),
                "case_count": case_count,
                "executed": executed,
                "successes": success_rows,
                "partials": partial_rows,
                "fails": fail_rows,
                "errors": error_rows,
                "mean_score": mean_score,
            }
        )
    return items


def read_suite_progress(suite_dir: str) -> Dict[str, Any]:
    """Read incremental progress from a running (or finished) suite.

    Returns {"total": N, "completed": M, "rows": [...]}.
    """
    root = Path(suite_dir)
    manifest = read_json(root / "manifest.json")
    total = int(manifest.get("case_count", 0))
    results = read_json(root / "results.json")
    rows = results.get("rows", [])
    rows = rows if isinstance(rows, list) else []
    return {"total": total, "completed": len(rows), "rows": rows}


def status_badge(status: str) -> str:
    """Format a status string as a badge label."""
    s = (status or "").lower()
    if s == "success":
        return "SUCCESS"
    if s == "partial":
        return "PARTIAL"
    if s == "fail":
        return "FAIL"
    if s == "running":
        return "RUNNING"
    return (status or "unknown").upper()


def artifact_files(run_dir: str) -> List[Path]:
    """List artifact files under run_dir/artifacts."""
    artifacts = Path(run_dir) / "artifacts"
    if not artifacts.exists():
        return []
    return [p for p in sorted(artifacts.rglob("*")) if p.is_file()]


def safe_float(value: Any) -> Optional[float]:
    """Convert to float or return None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_failure_modes(
    status: str,
    violation_code_counts: Dict[str, int],
    debug_payload: Dict[str, Any],
) -> List[str]:
    """Derive human-readable failure mode hints from run data."""
    modes: List[str] = []
    code_keys = set(violation_code_counts.keys())

    if status == "fail":
        if "no_candidates_generated" in code_keys:
            modes.append("No candidate planes/bends generated from mesh.")
        else:
            modes.append("Pipeline produced no valid final parts.")

    if any(code.startswith("dfm_sheet_size") for code in code_keys):
        modes.append("Parts exceed sheet-size constraints (scale/splitting issue).")
    if "unsupported_geometry" in code_keys:
        modes.append("Mesh contains geometry outside current flat-cut/bend capability.")
    if any(code.startswith("bend_") for code in code_keys):
        modes.append("Bend capability limits violated (angle/radius/material).")
    if any(code.startswith("min_feature") for code in code_keys):
        modes.append("Small features below manufacturing minimum.")

    dropped = int(debug_payload.get("intersection_dropped_count", 0) or 0)
    if dropped > 0:
        modes.append(f"Intersection filter removed {dropped} candidate layers/parts.")

    thin_side_dropped = int(debug_payload.get("thin_side_dropped_count", 0) or 0)
    if thin_side_dropped > 0:
        modes.append(f"Thin-side suppression dropped {thin_side_dropped} candidates.")

    if not modes and status in {"partial", "fail"}:
        modes.append("No dominant mode detected; inspect per-case violations/debug.")
    return modes


def suite_rows_by_case(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index suite result rows by case_id."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            out[case_id] = row
    return out
