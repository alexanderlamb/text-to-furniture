#!/usr/bin/env python3
"""Run hybrid fabrication composition from multiple strategy outputs."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fabrication.contracts import FabricationConfig
from fabrication.hybrid import run_hybrid_composition, write_hybrid_artifacts
from run_protocol import (
    copy_input_mesh,
    prepare_run_dir,
    update_latest_pointer,
    write_json,
    write_text,
)

HYBRID_ARTIFACT_DIRNAME = "hybrid_fabrication"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run hybrid fabrication composition over selected strategy outputs "
            "and write a composed artifact bundle."
        )
    )
    parser.add_argument(
        "--mesh", required=True, help="Path to input mesh (.stl/.obj/.ply/.glb)"
    )
    parser.add_argument("--name", default="hybrid_fabrication", help="Design/run name")
    parser.add_argument("--runs-dir", default="runs", help="Runs output root")
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        help="Source strategy id to include. May be repeated.",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated source strategy ids. Defaults to config strategies.",
    )
    parser.add_argument(
        "--material",
        "--material-key",
        dest="material_key",
        default="plywood_baltic_birch",
        help="Material key from materials.MATERIALS",
    )
    parser.add_argument(
        "--thickness-mm",
        type=float,
        default=None,
        help="Preferred material thickness in mm",
    )
    parser.add_argument(
        "--target-height-mm",
        type=float,
        default=750.0,
        help="Target Z height for mesh normalization",
    )
    parser.add_argument(
        "--no-auto-scale",
        action="store_true",
        help="Disable mesh normalization scaling",
    )
    parser.add_argument(
        "--part-budget",
        type=int,
        default=48,
        help="Maximum physical part count target passed to source strategies",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=6,
        help="Maximum number of regions to compose into the hybrid plan",
    )
    parser.add_argument(
        "--min-feature-mm",
        type=float,
        default=3.175,
        help="Minimum feature size in mm for DFM checks",
    )
    parser.add_argument(
        "--voxel-pitch-multiplier",
        type=float,
        default=4.0,
        help="Voxel strategy pitch as a multiple of selected material thickness",
    )
    parser.add_argument(
        "--max-voxels-per-axis",
        type=int,
        default=36,
        help="Voxel strategy resolution cap per axis",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logs"
    )
    return parser


def parse_strategy_ids(args: argparse.Namespace) -> tuple[str, ...] | None:
    ids: list[str] = []
    if args.strategies:
        ids.extend(
            part.strip()
            for part in args.strategies.split(",")
            if part.strip() and part.strip().lower() != "all"
        )
    ids.extend(
        strategy_id.strip() for strategy_id in args.strategy if strategy_id.strip()
    )

    deduped: list[str] = []
    seen: set[str] = set()
    for strategy_id in ids:
        if strategy_id not in seen:
            deduped.append(strategy_id)
            seen.add(strategy_id)
    return tuple(deduped) or None


def _build_config(
    args: argparse.Namespace,
    *,
    mesh_path: Path,
    strategy_ids: tuple[str, ...] | None,
) -> FabricationConfig:
    kwargs = {
        "mesh_path": str(mesh_path),
        "design_name": args.name,
        "material_key": args.material_key,
        "preferred_thickness_mm": args.thickness_mm,
        "auto_scale": not args.no_auto_scale,
        "target_height_mm": float(args.target_height_mm),
        "part_budget_max": max(1, int(args.part_budget)),
        "min_feature_mm": max(0.0, float(args.min_feature_mm)),
        "voxel_pitch_multiplier": max(0.001, float(args.voxel_pitch_multiplier)),
        "max_voxels_per_axis": max(1, int(args.max_voxels_per_axis)),
        "max_hybrid_regions": max(1, int(args.max_regions)),
    }
    if strategy_ids is not None:
        kwargs["strategies"] = strategy_ids
    return FabricationConfig(**kwargs)


def _build_summary(
    *,
    run_id: str,
    elapsed_s: float,
    strategy_ids: tuple[str, ...] | None,
    hybrid_plan_path: Path,
    status: str,
    region_count: int,
    assignment_count: int,
    part_count: int,
) -> str:
    strategies_text = ", ".join(strategy_ids) if strategy_ids else "config default"
    return "\n".join(
        [
            f"# Hybrid Fabrication {run_id}",
            "",
            f"- Status: **{status.upper()}**",
            f"- Duration: {elapsed_s:.2f}s",
            f"- Source strategies requested: {strategies_text}",
            f"- Regions: {region_count}",
            f"- Assignments: {assignment_count}",
            f"- Parts: {part_count}",
            f"- Hybrid plan: `{hybrid_plan_path}`",
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
    strategy_ids = parse_strategy_ids(args)
    run_paths = prepare_run_dir(args.runs_dir, args.name)
    copied_mesh = copy_input_mesh(args.mesh, run_paths.input_dir)
    hybrid_artifacts_dir = run_paths.artifacts_dir / HYBRID_ARTIFACT_DIRNAME

    config = _build_config(args, mesh_path=copied_mesh, strategy_ids=strategy_ids)
    result = run_hybrid_composition(
        config,
        artifacts_dir=hybrid_artifacts_dir,
        strategy_ids=strategy_ids,
        max_regions=config.max_hybrid_regions,
    )
    write_hybrid_artifacts(result, hybrid_artifacts_dir)
    elapsed = time.perf_counter() - started

    plan = result.get("hybrid_plan")
    status = str(getattr(plan, "status", "unknown"))
    region_count = len(getattr(plan, "regions", []))
    assignment_count = len(getattr(plan, "assignments", []))
    part_count = len(getattr(plan, "parts", []))
    joint_count = len(getattr(plan, "joints", []))
    operation_count = len(getattr(plan, "operations", []))
    overall_score = float(getattr(plan, "scores", {}).get("overall", 0.0))

    hybrid_plan_path = hybrid_artifacts_dir / "hybrid_plan.json"
    hybrid_summary_path = hybrid_artifacts_dir / "hybrid_summary.json"
    source_dir = hybrid_artifacts_dir / "source_strategies"
    source_ranking_path = source_dir / "ranking.json"
    source_artifacts = {}
    if isinstance(result.get("source_tournament"), dict):
        source_artifacts = {
            "source_strategies_dir": str(source_dir),
            "source_ranking": str(source_ranking_path),
        }
    requested_strategies = (
        list(strategy_ids) if strategy_ids else list(config.strategies)
    )

    metrics_payload = {
        "run_id": run_paths.run_id,
        "strategy": "hybrid_fabrication",
        "status": status,
        "elapsed_s": round(elapsed, 3),
        "requested_strategies": requested_strategies,
        "counts": {
            "regions": region_count,
            "assignments": assignment_count,
            "parts": part_count,
            "joints": joint_count,
            "operations": operation_count,
        },
        "scores": {"overall": overall_score},
        "artifacts": {
            "hybrid_fabrication_dir": str(hybrid_artifacts_dir),
            "hybrid_plan": str(hybrid_plan_path),
            "hybrid_summary": str(hybrid_summary_path),
            **source_artifacts,
        },
    }
    write_json(run_paths.metrics_path, metrics_payload)

    summary = _build_summary(
        run_id=run_paths.run_id,
        elapsed_s=elapsed,
        strategy_ids=strategy_ids,
        hybrid_plan_path=hybrid_plan_path,
        status=status,
        region_count=region_count,
        assignment_count=assignment_count,
        part_count=part_count,
    )
    write_text(run_paths.summary_path, summary)

    manifest = {
        "run_id": run_paths.run_id,
        "strategy": "hybrid_fabrication",
        "design_name": args.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_mesh": str(copied_mesh),
        "status": status,
        "config": {
            "mesh": str(copied_mesh),
            "material_key": config.material_key,
            "preferred_thickness_mm": config.preferred_thickness_mm,
            "auto_scale": config.auto_scale,
            "target_height_mm": config.target_height_mm,
            "part_budget_max": config.part_budget_max,
            "min_feature_mm": config.min_feature_mm,
            "voxel_pitch_multiplier": config.voxel_pitch_multiplier,
            "max_voxels_per_axis": config.max_voxels_per_axis,
            "max_hybrid_regions": config.max_hybrid_regions,
            "requested_strategies": requested_strategies,
        },
        "artifacts": {
            "hybrid_fabrication_dir": str(hybrid_artifacts_dir),
            "hybrid_plan": str(hybrid_plan_path),
            "hybrid_summary": str(hybrid_summary_path),
            **source_artifacts,
            "metrics": str(run_paths.metrics_path),
            "summary": str(run_paths.summary_path),
        },
    }
    write_json(run_paths.manifest_path, manifest)
    update_latest_pointer(args.runs_dir, run_paths.run_dir)

    print(f"Run ID: {run_paths.run_id}")
    print(f"Run dir: {run_paths.run_dir}")
    print(f"Status: {status.upper()}")
    print(f"Requested strategies: {requested_strategies}")
    print(f"Regions: {region_count}")
    print(f"Assignments: {assignment_count}")
    print(f"Parts: {part_count}")
    print(f"Hybrid plan: {hybrid_plan_path}")
    if source_ranking_path.exists():
        print(f"Source ranking: {source_ranking_path}")
    return 0 if status != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
