#!/usr/bin/env python3
"""Run a prototype multi-strategy fabrication tournament."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fabrication import (
    FabricationConfig,
    run_tournament,
    write_tournament_artifacts,
)
from run_protocol import (
    copy_input_mesh,
    prepare_run_dir,
    update_latest_pointer,
    write_json,
    write_text,
)

TOURNAMENT_ARTIFACT_DIRNAME = "fabrication_tournament"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected fabrication strategies against a mesh and write a "
            "ranked tournament artifact bundle."
        )
    )
    parser.add_argument(
        "--mesh", required=True, help="Path to input mesh (.stl/.obj/.ply/.glb)"
    )
    parser.add_argument(
        "--name", default="fabrication_tournament", help="Design/run name"
    )
    parser.add_argument("--runs-dir", default="runs", help="Runs output root")
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        help="Strategy id to run. May be repeated.",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated strategy ids to run. Defaults to config strategies.",
    )
    parser.add_argument(
        "--material-key",
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
        help="Maximum physical part count target passed to strategies",
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
    }
    if strategy_ids is not None:
        kwargs["strategies"] = strategy_ids
    return FabricationConfig(**kwargs)


def _build_summary(
    *,
    run_id: str,
    elapsed_s: float,
    strategy_ids: tuple[str, ...] | None,
    ranking_path: Path,
    plan_count: int,
) -> str:
    strategies_text = ", ".join(strategy_ids) if strategy_ids else "config default"
    return "\n".join(
        [
            f"# Fabrication Tournament {run_id}",
            "",
            f"- Status: **OK**",
            f"- Duration: {elapsed_s:.2f}s",
            f"- Strategies requested: {strategies_text}",
            f"- Plans emitted: {plan_count}",
            f"- Ranking: `{ranking_path}`",
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
    tournament_artifacts_dir = run_paths.artifacts_dir / TOURNAMENT_ARTIFACT_DIRNAME

    config = _build_config(args, mesh_path=copied_mesh, strategy_ids=strategy_ids)
    result = run_tournament(
        config,
        artifacts_dir=tournament_artifacts_dir,
        strategy_ids=strategy_ids,
    )
    write_tournament_artifacts(result, tournament_artifacts_dir)
    elapsed = time.perf_counter() - started

    ranking_path = tournament_artifacts_dir / "ranking.json"
    ranking = result.get("ranking", [])
    plans = result.get("plans", {})
    plan_count = len(plans) if isinstance(plans, dict) else 0

    metrics_payload = {
        "run_id": run_paths.run_id,
        "strategy": "fabrication_tournament",
        "status": "ok",
        "elapsed_s": round(elapsed, 3),
        "requested_strategies": (
            list(strategy_ids) if strategy_ids else list(config.strategies)
        ),
        "counts": {
            "plans": plan_count,
            "ranked_strategies": len(ranking) if isinstance(ranking, list) else 0,
        },
        "artifacts": {
            "fabrication_tournament_dir": str(tournament_artifacts_dir),
            "ranking": str(ranking_path),
        },
    }
    write_json(run_paths.metrics_path, metrics_payload)

    summary = _build_summary(
        run_id=run_paths.run_id,
        elapsed_s=elapsed,
        strategy_ids=strategy_ids,
        ranking_path=ranking_path,
        plan_count=plan_count,
    )
    write_text(run_paths.summary_path, summary)

    manifest = {
        "run_id": run_paths.run_id,
        "strategy": "fabrication_tournament",
        "design_name": args.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_mesh": str(copied_mesh),
        "status": "ok",
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
            "requested_strategies": (
                list(strategy_ids) if strategy_ids else list(config.strategies)
            ),
        },
        "artifacts": {
            "fabrication_tournament_dir": str(tournament_artifacts_dir),
            "ranking": str(ranking_path),
            "metrics": str(run_paths.metrics_path),
            "summary": str(run_paths.summary_path),
        },
    }
    write_json(run_paths.manifest_path, manifest)
    update_latest_pointer(args.runs_dir, run_paths.run_dir)

    print(f"Run ID: {run_paths.run_id}")
    print(f"Run dir: {run_paths.run_dir}")
    print("Status: OK")
    print(
        "Requested strategies: "
        f"{list(strategy_ids) if strategy_ids else list(config.strategies)}"
    )
    print(f"Plans: {plan_count}")
    print(f"Ranking: {ranking_path}")
    if isinstance(ranking, list) and ranking:
        winner = ranking[0]
        if isinstance(winner, dict):
            print(f"Winner: {winner.get('strategy_id', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
