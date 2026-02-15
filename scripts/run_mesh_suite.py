#!/usr/bin/env python3
"""Run a standard mesh suite and emit comparable progress metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from materials import MATERIALS
from pipeline import PipelineConfig, run_pipeline_from_mesh
from step3_first_principles import Step3Input, build_default_capability_profile

DEFAULT_SUITE_FILE = ROOT / "benchmarks" / "mesh_suite.json"
DEFAULT_RUNS_DIR = ROOT / "runs"
DEFAULT_SUITES_DIR = ROOT / "runs" / "suites"


@dataclass
class SuiteCase:
    case_id: str
    mesh_path: str


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "suite"


def _resolve_path(path_value: str, base: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def load_suite(path: Path) -> Tuple[str, str, List[SuiteCase]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    suite_name = str(payload.get("suite_name", "mesh_suite")).strip() or "mesh_suite"
    description = str(payload.get("description", "")).strip()
    raw_cases = payload.get("cases", [])
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("suite file must contain a non-empty 'cases' list")

    cases: List[SuiteCase] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("id", "")).strip()
        mesh_path = str(item.get("mesh_path", "")).strip()
        if not case_id or not mesh_path:
            continue
        cases.append(SuiteCase(case_id=case_id, mesh_path=mesh_path))

    if not cases:
        raise ValueError("suite file contained no valid cases")
    return suite_name, description, cases


def _count_violations(violations: List[Dict[str, Any]], severity: str) -> int:
    return sum(1 for v in violations if str(v.get("severity", "")).lower() == severity)


def _load_previous_results(suites_dir: Path) -> Tuple[Optional[str], Dict[str, Dict[str, Any]]]:
    latest = suites_dir / "latest"
    if latest.is_symlink():
        target = (suites_dir / os.readlink(latest)).resolve()
        prev_path = target / "results.json"
    else:
        prev_path = latest / "results.json"

    if not prev_path.exists():
        return None, {}

    payload = json.loads(prev_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("case_id", "")).strip()
        if key:
            out[key] = row
    return str(prev_path.parent.name), out


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_step3_overrides(raw: str) -> Dict[str, Any]:
    if not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--step3-overrides-json must decode to an object")
    return parsed


def _apply_step3_overrides(step3_input: Step3Input, overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(step3_input, key):
            continue
        if key in {"mesh_path", "design_name"}:
            continue
        setattr(step3_input, key, value)


def _violation_code_counts(violations: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for v in violations:
        code = str(v.get("code", "")).strip()
        if not code:
            continue
        counts[code] = counts.get(code, 0) + 1
    return counts


def _dominant_failure_mode(
    status: str,
    violation_code_counts: Dict[str, int],
    debug: Dict[str, Any],
) -> str:
    if status == "fail":
        if "no_candidates_generated" in violation_code_counts:
            return "no_candidates_generated"
        return "pipeline_failed"
    if any(code.startswith("dfm_sheet_size") for code in violation_code_counts):
        return "sheet_size_violation"
    if "unsupported_geometry" in violation_code_counts:
        return "unsupported_geometry"
    if any(code.startswith("bend_") for code in violation_code_counts):
        return "bend_limits"
    if int(debug.get("intersection_dropped_count", 0) or 0) > 0:
        return "intersection_filter_pruning"
    if int(debug.get("thin_side_dropped_count", 0) or 0) > 0:
        return "thin_side_suppression"
    if int(sum(violation_code_counts.values())) > 0:
        return "mixed_violations"
    return "ok"


def _build_summary_markdown(
    suite_name: str,
    description: str,
    suite_run_id: str,
    rows: List[Dict[str, Any]],
    previous_run_id: Optional[str],
    previous_by_case: Dict[str, Dict[str, Any]],
) -> str:
    run_rows = [r for r in rows if r.get("status") not in {"missing_input", "error"}]
    score_vals = [r["overall_score"] for r in run_rows if isinstance(r.get("overall_score"), float)]
    haus_vals = [r["hausdorff_mm"] for r in run_rows if isinstance(r.get("hausdorff_mm"), float)]
    diff_vals = [
        r["plane_constraint_difficulty_weighted"]
        for r in run_rows
        if isinstance(r.get("plane_constraint_difficulty_weighted"), float)
    ]

    lines = [
        f"# Suite {suite_run_id}",
        "",
        f"- Suite name: `{suite_name}`",
        f"- Description: {description or 'n/a'}",
        f"- Cases: {len(rows)}",
        f"- Executed: {len(run_rows)}",
        f"- Missing input: {sum(1 for r in rows if r.get('status') == 'missing_input')}",
        f"- Errors: {sum(1 for r in rows if r.get('status') == 'error')}",
        f"- Mean score: {sum(score_vals) / len(score_vals):.3f}" if score_vals else "- Mean score: n/a",
        f"- Mean hausdorff (mm): {sum(haus_vals) / len(haus_vals):.2f}" if haus_vals else "- Mean hausdorff (mm): n/a",
        (
            f"- Mean weighted plane difficulty: {sum(diff_vals) / len(diff_vals):.3f}"
            if diff_vals
            else "- Mean weighted plane difficulty: n/a"
        ),
    ]

    if previous_run_id:
        lines.extend(["", f"## Delta vs `{previous_run_id}`"])
        deltas: List[str] = []
        for row in run_rows:
            case_id = str(row.get("case_id", ""))
            prev = previous_by_case.get(case_id)
            if not prev:
                continue
            new_score = _safe_float(row.get("overall_score"))
            old_score = _safe_float(prev.get("overall_score"))
            new_diff = _safe_float(row.get("plane_constraint_difficulty_weighted"))
            old_diff = _safe_float(prev.get("plane_constraint_difficulty_weighted"))
            if new_score is None or old_score is None:
                continue
            score_delta = new_score - old_score
            if new_diff is not None and old_diff is not None:
                diff_delta = new_diff - old_diff
                deltas.append(
                    f"- `{case_id}` score {score_delta:+.3f}, difficulty {diff_delta:+.3f}"
                )
            else:
                deltas.append(f"- `{case_id}` score {score_delta:+.3f}")
        lines.extend(deltas or ["- No comparable previous case data"])

    lines.extend(["", "## Case Results"])
    for row in rows:
        case_id = str(row.get("case_id", "case"))
        status = str(row.get("status", "unknown")).upper()
        score = row.get("overall_score")
        score_text = f"{score:.3f}" if isinstance(score, float) else "n/a"
        haus = row.get("hausdorff_mm")
        haus_text = f"{haus:.2f}" if isinstance(haus, float) else "n/a"
        diff = row.get("plane_constraint_difficulty_weighted")
        diff_text = f"{diff:.3f}" if isinstance(diff, float) else "n/a"
        errs = int(row.get("errors", 0) or 0)
        warns = int(row.get("warnings", 0) or 0)
        lines.append(
            f"- `{case_id}` status={status}, score={score_text}, hausdorff={haus_text}mm, "
            f"difficulty={diff_text}, violations={errs}e/{warns}w"
        )
    return "\n".join(lines) + "\n"


def _update_latest_pointer(suites_dir: Path, suite_dir: Path) -> None:
    latest = suites_dir / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            for child in latest.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
            latest.rmdir()
    try:
        latest.symlink_to(os.path.relpath(suite_dir, suites_dir))
    except OSError:
        latest.mkdir(parents=True, exist_ok=True)
        (latest / "latest_suite.txt").write_text(suite_dir.name + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standard mesh benchmark suite")
    parser.add_argument("--suite-file", default=str(DEFAULT_SUITE_FILE))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--suites-dir", default=str(DEFAULT_SUITES_DIR))
    parser.add_argument(
        "--material",
        default="plywood_baltic_birch",
        choices=sorted(MATERIALS.keys()),
    )
    parser.add_argument("--part-budget", type=int, default=10)
    parser.add_argument("--fidelity-weight", type=float, default=0.75)
    parser.add_argument("--target-height-mm", type=float, default=750.0)
    parser.add_argument("--no-auto-scale", action="store_true")
    parser.add_argument("--no-bending", action="store_true")
    parser.add_argument("--with-exports", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--strict-missing", action="store_true")
    parser.add_argument(
        "--step3-overrides-json",
        default="",
        help="JSON object of Step3Input field overrides (applied after CLI defaults).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    step3_overrides = _parse_step3_overrides(args.step3_overrides_json)
    suite_file = _resolve_path(args.suite_file, ROOT)
    suite_name, description, cases = load_suite(suite_file)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    runs_dir = _resolve_path(args.runs_dir, ROOT)
    suites_dir = _resolve_path(args.suites_dir, ROOT)
    runs_dir.mkdir(parents=True, exist_ok=True)
    suites_dir.mkdir(parents=True, exist_ok=True)

    previous_run_id, previous_by_case = _load_previous_results(suites_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_run_id = f"{stamp}_{slugify(suite_name)}"
    suite_dir = suites_dir / suite_run_id
    suite_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "suite_run_id": suite_run_id,
        "suite_name": suite_name,
        "description": description,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "suite_file": str(suite_file),
        "runs_dir": str(runs_dir),
        "args": vars(args),
        "step3_overrides": step3_overrides,
        "case_count": len(cases),
    }
    (suite_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    rows: List[Dict[str, Any]] = []
    for case in cases:
        mesh_path = _resolve_path(case.mesh_path, ROOT)
        if not mesh_path.is_file():
            msg = f"missing mesh: {mesh_path}"
            if args.strict_missing:
                raise FileNotFoundError(msg)
            rows.append(
                {
                    "case_id": case.case_id,
                    "mesh_path": str(mesh_path),
                    "status": "missing_input",
                    "error": msg,
                }
            )
            print(f"[skip] {case.case_id}: {msg}")
            continue

        capability = build_default_capability_profile(
            material_key=args.material,
            allow_controlled_bending=not args.no_bending,
        )
        step3_input = Step3Input(
            mesh_path=str(mesh_path),
            design_name=case.case_id,
            fidelity_weight=max(0.0, min(1.0, args.fidelity_weight)),
            part_budget_max=max(1, int(args.part_budget)),
            material_preferences=[args.material],
            scs_capabilities=capability,
            target_height_mm=float(args.target_height_mm),
            auto_scale=not args.no_auto_scale,
        )
        _apply_step3_overrides(step3_input, step3_overrides)
        config = PipelineConfig(
            runs_dir=str(runs_dir),
            export_svg=bool(args.with_exports),
            export_nested_svg=bool(args.with_exports),
            export_dxf=bool(args.with_exports),
            export_nested_dxf=bool(args.with_exports),
            step3_input=step3_input,
        )

        try:
            result = run_pipeline_from_mesh(
                mesh_path=str(mesh_path),
                design_name=case.case_id,
                config=config,
            )
            metrics_payload = json.loads(
                Path(result.metrics_path).read_text(encoding="utf-8")
            )
            quality = metrics_payload.get("quality_metrics", {})
            counts = metrics_payload.get("counts", {})
            debug = metrics_payload.get("debug", {})
            violations = metrics_payload.get("violations", [])
            code_counts = _violation_code_counts(violations)
            dominant_mode = _dominant_failure_mode(
                status=str(metrics_payload.get("status", "unknown")).lower(),
                violation_code_counts=code_counts,
                debug=debug if isinstance(debug, dict) else {},
            )

            row = {
                "case_id": case.case_id,
                "mesh_path": str(mesh_path),
                "run_id": result.run_id,
                "run_dir": result.run_dir,
                "status": str(metrics_payload.get("status", "unknown")),
                "elapsed_s": _safe_float(metrics_payload.get("elapsed_s")),
                "overall_score": _safe_float(quality.get("overall_score")),
                "hausdorff_mm": _safe_float(quality.get("hausdorff_mm")),
                "mean_distance_mm": _safe_float(quality.get("mean_distance_mm")),
                "normal_error_deg": _safe_float(quality.get("normal_error_deg")),
                "part_count": int(counts.get("components", 0)),
                "joint_count": int(counts.get("joints", 0)),
                "errors": _count_violations(violations, "error"),
                "warnings": _count_violations(violations, "warning"),
                "coverage_ratio_unique_faces": _safe_float(
                    debug.get("coverage_ratio_unique_faces")
                ),
                "coverage_ratio_summed_source_area": _safe_float(
                    debug.get("coverage_ratio_summed_source_area")
                ),
                "plane_constraint_difficulty_mean": _safe_float(
                    debug.get("plane_constraint_difficulty_mean")
                ),
                "plane_constraint_difficulty_weighted": _safe_float(
                    debug.get("plane_constraint_difficulty_weighted")
                ),
                "plane_constraint_difficulty_max": _safe_float(
                    debug.get("plane_constraint_difficulty_max")
                ),
                "mesh_constraint_pressure_mean": _safe_float(
                    debug.get("mesh_constraint_pressure_mean")
                ),
                "intersection_constraint_pressure_mean": _safe_float(
                    debug.get("intersection_constraint_pressure_mean")
                ),
                "intersection_dropped_count": int(
                    debug.get("intersection_dropped_count", 0)
                ),
                "stacked_extra_layers": int(debug.get("stacked_extra_layers", 0)),
                "violation_code_counts": code_counts,
                "violation_codes": sorted(code_counts.keys()),
                "dominant_failure_mode": dominant_mode,
            }
            rows.append(row)
            print(
                f"[ok] {case.case_id}: status={row['status']} score={row['overall_score']}"
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            rows.append(
                {
                    "case_id": case.case_id,
                    "mesh_path": str(mesh_path),
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"[error] {case.case_id}: {exc}")

        # Flush incremental results so the UI can poll progress
        (suite_dir / "results.json").write_text(
            json.dumps({"suite_run_id": suite_run_id, "suite_name": suite_name, "rows": rows}, indent=2),
            encoding="utf-8",
        )

    summary_md = _build_summary_markdown(
        suite_name=suite_name,
        description=description,
        suite_run_id=suite_run_id,
        rows=rows,
        previous_run_id=previous_run_id,
        previous_by_case=previous_by_case,
    )

    results_payload = {
        "suite_run_id": suite_run_id,
        "suite_name": suite_name,
        "rows": rows,
    }

    (suite_dir / "results.json").write_text(
        json.dumps(results_payload, indent=2), encoding="utf-8"
    )
    (suite_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    fieldnames: List[str] = sorted(
        {key for row in rows for key in row.keys()},
    )
    with (suite_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _update_latest_pointer(suites_dir, suite_dir)

    print(f"Suite run: {suite_run_id}")
    print(f"Suite dir: {suite_dir}")
    print(f"Summary: {suite_dir / 'summary.md'}")
    print(f"Results: {suite_dir / 'results.json'}")

    has_errors = any(r.get("status") == "error" for r in rows)
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
