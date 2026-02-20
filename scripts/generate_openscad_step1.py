#!/usr/bin/env python3
"""Run clean-slate Step 1 (mesh -> parametric OpenSCAD panels)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openscad_step1 import Step1Config, run_step1_pipeline
from openscad_step1.audit import AuditTrail
from run_protocol import (
    copy_input_mesh,
    prepare_run_dir,
    update_latest_pointer,
    write_json,
    write_text,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean-slate Step 1: fill mesh with parametric OpenSCAD panels"
    )
    parser.add_argument(
        "--mesh", required=True, help="Path to input mesh (.stl/.obj/.ply/.glb)"
    )
    parser.add_argument("--name", default="openscad_step1", help="Design/run name")
    parser.add_argument("--runs-dir", default="runs", help="Runs output root")
    parser.add_argument(
        "--material-key",
        default="plywood_baltic_birch",
        help="Material key from materials.MATERIALS",
    )
    parser.add_argument(
        "--thickness-mm",
        type=float,
        default=None,
        help="Preferred material thickness in mm (nearest available is used)",
    )
    parser.add_argument(
        "--target-height-mm",
        type=float,
        default=750.0,
        help="Target Z height for auto scaling",
    )
    parser.add_argument(
        "--no-auto-scale", action="store_true", help="Disable normalization scaling"
    )
    parser.add_argument(
        "--part-budget",
        type=int,
        default=18,
        help="Maximum physical panel instances selected",
    )
    parser.add_argument(
        "--min-region-area-mm2",
        type=float,
        default=450.0,
        help="Minimum planar region area considered for candidates",
    )
    parser.add_argument(
        "--max-stack-layers",
        type=int,
        default=6,
        help="Maximum stacked layers inferred from opposite planes",
    )
    parser.add_argument(
        "--stack-roundup-bias",
        type=float,
        default=0.35,
        help="Rounding bias for stack layer quantization (0-1)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logs"
    )
    return parser


def _build_summary(
    *,
    run_id: str,
    elapsed_s: float,
    status: str,
    panel_count: int,
    family_count: int,
    selected_family_count: int,
    violations_error: int,
    violations_warning: int,
) -> str:
    return "\n".join(
        [
            f"# Run {run_id}",
            "",
            f"- Status: **{status.upper()}**",
            f"- Duration: {elapsed_s:.2f}s",
            f"- Panels: {panel_count}",
            f"- Families: {selected_family_count}/{family_count} selected",
            f"- Violations: {violations_error} error, {violations_warning} warning",
            "",
            "## Scope",
            "- Step 1 only: mesh -> parametric OpenSCAD panelization",
            "- No cut-file generation in this command",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    started = time.perf_counter()
    run_paths = prepare_run_dir(args.runs_dir, args.name)
    copied_mesh = copy_input_mesh(args.mesh, run_paths.input_dir)

    config = Step1Config(
        mesh_path=str(copied_mesh),
        design_name=args.name,
        material_key=args.material_key,
        preferred_thickness_mm=args.thickness_mm,
        auto_scale=not args.no_auto_scale,
        target_height_mm=float(args.target_height_mm),
        part_budget_max=max(1, int(args.part_budget)),
        min_region_area_mm2=max(1.0, float(args.min_region_area_mm2)),
        max_stack_layers=max(1, int(args.max_stack_layers)),
        stack_roundup_bias=max(0.0, min(1.0, float(args.stack_roundup_bias))),
    )

    audit = AuditTrail(run_id=run_paths.run_id, artifacts_dir=run_paths.artifacts_dir)
    result = run_step1_pipeline(
        config=config,
        run_id=run_paths.run_id,
        artifacts_dir=run_paths.artifacts_dir,
        audit=audit,
    )
    elapsed = time.perf_counter() - started

    design_path = run_paths.artifacts_dir / "design_step1_openscad.json"
    scad_path = run_paths.artifacts_dir / "model_step1.scad"
    capsule_path = run_paths.artifacts_dir / "spatial_capsule_step1.json"

    write_json(design_path, result.design_payload)
    write_text(scad_path, result.openscad_code)
    write_json(capsule_path, result.spatial_capsule)

    errors = sum(1 for violation in result.violations if violation.severity == "error")
    warnings = sum(
        1 for violation in result.violations if violation.severity == "warning"
    )
    metrics_payload = {
        "run_id": result.run_id,
        "strategy": "openscad_step1_clean_slate",
        "status": result.status,
        "elapsed_s": round(elapsed, 3),
        "mesh_hash_sha256": result.mesh_hash_sha256,
        "material_key": result.material_key,
        "material_thickness_mm": result.material_thickness_mm,
        "mesh_bounds_mm": result.mesh_bounds_mm,
        "scale_factor": result.scale_factor,
        "counts": {
            "panel_families": len(result.panel_families),
            "selected_families": len(result.selected_families),
            "panels": len(result.panels),
            "violations": len(result.violations),
            "violations_error": errors,
            "violations_warning": warnings,
            "trim_pairs": len(result.trim_decisions),
        },
        "debug": result.debug,
    }
    write_json(run_paths.metrics_path, metrics_payload)

    summary = _build_summary(
        run_id=result.run_id,
        elapsed_s=elapsed,
        status=result.status,
        panel_count=len(result.panels),
        family_count=len(result.panel_families),
        selected_family_count=len(result.selected_families),
        violations_error=errors,
        violations_warning=warnings,
    )
    write_text(run_paths.summary_path, summary)

    manifest = {
        "run_id": result.run_id,
        "strategy": "openscad_step1_clean_slate",
        "design_name": args.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_mesh": str(copied_mesh),
        "status": result.status,
        "config": {
            "mesh": str(copied_mesh),
            "material_key": config.material_key,
            "preferred_thickness_mm": config.preferred_thickness_mm,
            "auto_scale": config.auto_scale,
            "target_height_mm": config.target_height_mm,
            "part_budget_max": config.part_budget_max,
            "min_region_area_mm2": config.min_region_area_mm2,
            "max_stack_layers": config.max_stack_layers,
            "stack_roundup_bias": config.stack_roundup_bias,
            "selection_mode": config.selection_mode,
            "cavity_axis_policy": config.cavity_axis_policy,
            "shell_policy": config.shell_policy,
            "overlap_enforcement": config.overlap_enforcement,
            "thin_gap_clearance_mm": config.thin_gap_clearance_mm,
        },
        "artifacts": {
            "design_json": str(design_path),
            "openscad_code": str(scad_path),
            "spatial_capsule": str(capsule_path),
            "metrics": str(run_paths.metrics_path),
            "summary": str(run_paths.summary_path),
            "checkpoints": [str(path) for path in result.checkpoints],
            "decision_log": str(result.decision_log_path),
            "decision_hash_chain": str(result.decision_hash_chain_path),
        },
    }
    write_json(run_paths.manifest_path, manifest)
    update_latest_pointer(args.runs_dir, run_paths.run_dir)

    print(f"Run ID: {result.run_id}")
    print(f"Run dir: {run_paths.run_dir}")
    print(f"Status: {result.status.upper()}")
    print(f"Panel families: {len(result.panel_families)}")
    print(f"Selected families: {len(result.selected_families)}")
    print(f"Panels: {len(result.panels)}")
    print(f"Violations: {errors} errors, {warnings} warnings")
    print(f"OpenSCAD: {scad_path}")
    print(f"Design JSON: {design_path}")
    print(f"Spatial capsule: {capsule_path}")
    print(f"Decision log: {result.decision_log_path}")
    print(f"Metrics: {run_paths.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
