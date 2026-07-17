#!/usr/bin/env python3
"""Run hybrid fabrication over benchmark meshes and aggregate evaluation metrics."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fabrication.contracts import FabricationConfig, HybridFabricationPlan
from fabrication.hybrid import run_hybrid_composition, write_hybrid_artifacts
from fabrication.visual_report import write_evaluation_report
from run_protocol import (
    prepare_run_dir,
    update_latest_pointer,
    write_json,
    write_text,
)

HYBRID_EVAL_DIRNAME = "hybrid_evaluation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid fabrication composition across benchmark meshes."
    )
    parser.add_argument(
        "--mesh",
        action="append",
        default=[],
        help="Mesh path to evaluate. May be repeated.",
    )
    parser.add_argument(
        "--mesh-dir",
        default="benchmarks/meshes",
        help="Directory of .stl meshes used when --mesh is omitted.",
    )
    parser.add_argument("--name", default="hybrid-benchmark-eval", help="Run name")
    parser.add_argument("--runs-dir", default="runs", help="Runs output root")
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated strategy ids. Defaults to FabricationConfig strategies.",
    )
    parser.add_argument("--material-key", default="plywood_baltic_birch")
    parser.add_argument("--thickness-mm", type=float, default=None)
    parser.add_argument("--target-height-mm", type=float, default=750.0)
    parser.add_argument("--no-auto-scale", action="store_true")
    parser.add_argument("--part-budget", type=int, default=48)
    parser.add_argument("--max-regions", type=int, default=6)
    parser.add_argument("--min-feature-mm", type=float, default=3.175)
    parser.add_argument("--voxel-pitch-multiplier", type=float, default=4.0)
    parser.add_argument("--max-voxels-per-axis", type=int, default=36)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first mesh exception instead of recording an error row.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    mesh_paths = _resolve_mesh_paths(args)
    if not mesh_paths:
        parser.error("No meshes found to evaluate.")

    started = time.perf_counter()
    run_paths = prepare_run_dir(args.runs_dir, args.name)
    eval_dir = run_paths.artifacts_dir / HYBRID_EVAL_DIRNAME
    eval_dir.mkdir(parents=True, exist_ok=True)
    strategy_ids = _parse_strategy_ids(args.strategies)

    rows: list[dict[str, object]] = []
    for mesh_path in mesh_paths:
        mesh_started = time.perf_counter()
        mesh_id = mesh_path.stem
        mesh_artifacts = eval_dir / mesh_id
        config = _build_config(args, mesh_path=mesh_path, strategy_ids=strategy_ids)

        try:
            result = run_hybrid_composition(
                config,
                artifacts_dir=mesh_artifacts,
                strategy_ids=strategy_ids,
                max_regions=config.max_hybrid_regions,
            )
            write_hybrid_artifacts(result, mesh_artifacts)
            plan = result["hybrid_plan"]
            assert isinstance(plan, HybridFabricationPlan)
            row = _row_from_plan(
                mesh_path=mesh_path,
                plan=plan,
                result=result,
                elapsed_s=time.perf_counter() - mesh_started,
                artifacts_dir=mesh_artifacts,
            )
        except Exception as exc:
            if args.fail_fast:
                raise
            row = _error_row(
                mesh_path=mesh_path,
                exc=exc,
                elapsed_s=time.perf_counter() - mesh_started,
                artifacts_dir=mesh_artifacts,
            )
        rows.append(row)
        _print_row(row)

    evaluation = _build_evaluation_payload(
        rows=rows,
        mesh_paths=mesh_paths,
        elapsed_s=time.perf_counter() - started,
        strategy_ids=strategy_ids,
        eval_dir=eval_dir,
    )
    eval_json = eval_dir / "evaluation.json"
    eval_csv = eval_dir / "evaluation.csv"
    report_html = eval_dir / "report.html"
    write_json(eval_json, evaluation)
    _write_csv(eval_csv, rows)
    write_evaluation_report(evaluation, report_html)

    write_json(
        run_paths.metrics_path,
        {
            "run_id": run_paths.run_id,
            "strategy": "hybrid_benchmark_evaluation",
            "status": evaluation["status"],
            "elapsed_s": evaluation["elapsed_s"],
            "counts": evaluation["counts"],
            "artifacts": {
                "evaluation_dir": str(eval_dir),
                "evaluation_json": str(eval_json),
                "evaluation_csv": str(eval_csv),
                "report_html": str(report_html),
            },
        },
    )
    write_json(
        run_paths.manifest_path,
        {
            "run_id": run_paths.run_id,
            "strategy": "hybrid_benchmark_evaluation",
            "design_name": args.name,
            "status": evaluation["status"],
            "mesh_count": len(mesh_paths),
            "config": {
                "mesh_paths": [str(path) for path in mesh_paths],
                "material_key": args.material_key,
                "preferred_thickness_mm": args.thickness_mm,
                "auto_scale": not args.no_auto_scale,
                "target_height_mm": args.target_height_mm,
                "part_budget_max": args.part_budget,
                "max_hybrid_regions": args.max_regions,
                "strategies": list(strategy_ids) if strategy_ids else None,
            },
            "artifacts": {
                "evaluation_dir": str(eval_dir),
                "evaluation_json": str(eval_json),
                "evaluation_csv": str(eval_csv),
                "report_html": str(report_html),
                "metrics": str(run_paths.metrics_path),
                "summary": str(run_paths.summary_path),
            },
        },
    )
    write_text(
        run_paths.summary_path,
        _summary_markdown(evaluation, eval_json, eval_csv, report_html),
    )
    update_latest_pointer(args.runs_dir, run_paths.run_dir)

    print("")
    print(f"Run ID: {run_paths.run_id}")
    print(f"Run dir: {run_paths.run_dir}")
    print(f"Evaluation JSON: {eval_json}")
    print(f"Evaluation CSV: {eval_csv}")
    print(f"Report HTML: {report_html}")
    print(f"Status: {evaluation['status'].upper()}")
    return 0 if evaluation["status"] != "error" else 1


def _resolve_mesh_paths(args: argparse.Namespace) -> list[Path]:
    if args.mesh:
        return sorted(Path(path) for path in args.mesh)
    mesh_dir = Path(args.mesh_dir)
    return sorted(mesh_dir.glob("*.stl"))


def _parse_strategy_ids(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    ids = tuple(part.strip() for part in value.split(",") if part.strip())
    return ids or None


def _build_config(
    args: argparse.Namespace,
    *,
    mesh_path: Path,
    strategy_ids: tuple[str, ...] | None,
) -> FabricationConfig:
    kwargs = {
        "mesh_path": str(mesh_path),
        "design_name": mesh_path.stem,
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


def _row_from_plan(
    *,
    mesh_path: Path,
    plan: HybridFabricationPlan,
    result: dict[str, object],
    elapsed_s: float,
    artifacts_dir: Path,
) -> dict[str, object]:
    strategy_mix = plan.debug.get("strategy_mix", {})
    if not isinstance(strategy_mix, dict):
        strategy_mix = {}
    source_tournament = result.get("source_tournament", {})
    source_ranking = []
    if isinstance(source_tournament, dict):
        source_ranking = source_tournament.get("ranking", [])
    source_winner = ""
    if isinstance(source_ranking, list) and source_ranking:
        first = source_ranking[0]
        if isinstance(first, dict):
            source_winner = str(first.get("strategy_id", ""))

    part_burden = _strategy_part_burden(plan)
    flags = _quality_flags(plan, strategy_mix, part_burden)
    return {
        "mesh": mesh_path.name,
        "mesh_path": str(mesh_path),
        "status": plan.status,
        "overall_score": float(plan.scores.get("overall", 0.0)),
        "regions": len(plan.regions),
        "assignments": len(plan.assignments),
        "parts": len(plan.parts),
        "parts_per_region": round(
            len(plan.parts) / max(float(len(plan.regions)), 1.0), 3
        ),
        "joints": len(plan.joints),
        "operations": len(plan.operations),
        "strategy_diversity": len(strategy_mix),
        "strategy_mix": dict(strategy_mix),
        "strategy_part_burden": part_burden["parts"],
        "strategy_volume_burden_mm3": part_burden["volume_mm3"],
        "strategy_area_burden_mm2": part_burden["area_mm2"],
        "dominant_part_strategy": _dominant_key(part_burden["parts"]),
        "source_part_reuse_count": int(plan.debug.get("source_part_reuse_count", 0)),
        "source_part_shared_count": int(plan.debug.get("source_part_shared_count", 0)),
        "source_winner": source_winner,
        "warning_count": len(plan.warnings),
        "warnings": list(plan.warnings),
        "flags": flags,
        "elapsed_s": round(float(elapsed_s), 3),
        "hybrid_plan": str(artifacts_dir / "hybrid_plan.json"),
        "source_ranking": str(artifacts_dir / "source_strategies" / "ranking.json"),
    }


def _error_row(
    *,
    mesh_path: Path,
    exc: Exception,
    elapsed_s: float,
    artifacts_dir: Path,
) -> dict[str, object]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifacts_dir / "exception.txt"
    trace_path.write_text(traceback.format_exc(), encoding="utf-8")
    return {
        "mesh": mesh_path.name,
        "mesh_path": str(mesh_path),
        "status": "error",
        "overall_score": 0.0,
        "regions": 0,
        "assignments": 0,
        "parts": 0,
        "joints": 0,
        "operations": 0,
        "strategy_diversity": 0,
        "strategy_mix": {},
        "strategy_part_burden": {},
        "strategy_volume_burden_mm3": {},
        "strategy_area_burden_mm2": {},
        "dominant_part_strategy": "",
        "source_part_reuse_count": 0,
        "source_part_shared_count": 0,
        "source_winner": "",
        "warning_count": 1,
        "warnings": [f"{type(exc).__name__}: {exc}"],
        "flags": ["exception"],
        "elapsed_s": round(float(elapsed_s), 3),
        "hybrid_plan": "",
        "source_ranking": "",
    }


def _quality_flags(
    plan: HybridFabricationPlan,
    strategy_mix: dict[str, object],
    part_burden: dict[str, dict[str, object]],
) -> list[str]:
    flags: list[str] = []
    part_count = len(plan.parts)
    region_count = len(plan.regions)
    assignment_count = len(plan.assignments)
    parts_per_region = part_count / max(float(region_count), 1.0)

    if plan.status == "error":
        flags.append("hybrid_error")
    if len(strategy_mix) < 2:
        flags.append("single_strategy_assignment")
    if assignment_count != region_count:
        flags.append("unassigned_regions")
    if not plan.joints and assignment_count > 1:
        flags.append("no_boundary_joints")
    if part_count > 96:
        flags.append("high_part_count")
    if parts_per_region > 8.0:
        flags.append("high_parts_per_region")
    if assignment_count > 0 and part_count < assignment_count * 1.25:
        flags.append("possibly_underbuilt")
    if assignment_count > 1 and len(plan.joints) < assignment_count - 1:
        flags.append("weak_boundary_density")
    if int(plan.debug.get("source_part_reuse_count", 0)) > 0:
        flags.append("duplicated_source_parts")
    strategy_part_burden = part_burden.get("parts", {})
    if len(strategy_part_burden) >= 2:
        burden_values = [int(value) for value in strategy_part_burden.values()]
        burden_total = sum(burden_values)
        if (
            burden_total >= 12
            and max(burden_values) >= 10
            and max(burden_values) / burden_total >= 0.72
        ):
            flags.append("dominant_strategy_burden")
    if float(plan.scores.get("overall", 0.0)) < 0.70:
        flags.append("low_score")
    if plan.warnings:
        flags.append("warnings_present")
    return flags


def _strategy_part_burden(plan: HybridFabricationPlan) -> dict[str, dict[str, object]]:
    part_counts: dict[str, int] = {}
    volume_mm3: dict[str, float] = {}
    area_mm2: dict[str, float] = {}
    for part in plan.parts:
        strategy_id = str(
            part.metadata.get("source_strategy_id") or part.strategy_id or "unknown"
        )
        quantity = _part_quantity(part.quantity)
        part_counts[strategy_id] = part_counts.get(strategy_id, 0) + quantity
        volume_mm3[strategy_id] = volume_mm3.get(strategy_id, 0.0) + (
            float(part.volume_mm3) * quantity
        )
        area_mm2[strategy_id] = area_mm2.get(strategy_id, 0.0) + (
            float(part.area_mm2) * quantity
        )

    return {
        "parts": dict(sorted(part_counts.items())),
        "volume_mm3": {
            strategy_id: round(value, 3)
            for strategy_id, value in sorted(volume_mm3.items())
        },
        "area_mm2": {
            strategy_id: round(value, 3)
            for strategy_id, value in sorted(area_mm2.items())
        },
    }


def _part_quantity(value: object) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _dominant_key(mapping: dict[str, object]) -> str:
    if not mapping:
        return ""
    return str(
        sorted(
            mapping.items(),
            key=lambda item: (-float(item[1]), str(item[0])),
        )[
            0
        ][0]
    )


def _build_evaluation_payload(
    *,
    rows: list[dict[str, object]],
    mesh_paths: list[Path],
    elapsed_s: float,
    strategy_ids: tuple[str, ...] | None,
    eval_dir: Path,
) -> dict[str, object]:
    errors = sum(1 for row in rows if row["status"] == "error")
    mixed = sum(1 for row in rows if int(row["strategy_diversity"]) >= 2)
    with_joints = sum(1 for row in rows if int(row["joints"]) > 0)
    total_flags: dict[str, int] = {}
    strategy_use: dict[str, int] = {}
    strategy_part_use: dict[str, int] = {}
    strategy_volume_use: dict[str, float] = {}
    strategy_area_use: dict[str, float] = {}
    for row in rows:
        for flag in row["flags"]:
            total_flags[str(flag)] = total_flags.get(str(flag), 0) + 1
        mix = row.get("strategy_mix", {})
        if isinstance(mix, dict):
            for strategy_id, count in mix.items():
                strategy_use[str(strategy_id)] = strategy_use.get(
                    str(strategy_id), 0
                ) + int(count)
        _merge_int_mapping(strategy_part_use, row.get("strategy_part_burden", {}))
        _merge_float_mapping(
            strategy_volume_use, row.get("strategy_volume_burden_mm3", {})
        )
        _merge_float_mapping(strategy_area_use, row.get("strategy_area_burden_mm2", {}))

    status = (
        "error" if errors == len(rows) else "warning" if errors or total_flags else "ok"
    )
    return {
        "schema_version": "fabrication.hybrid_evaluation.v0",
        "status": status,
        "elapsed_s": round(float(elapsed_s), 3),
        "strategies": list(strategy_ids) if strategy_ids else "config_default",
        "mesh_paths": [str(path) for path in mesh_paths],
        "counts": {
            "meshes": len(rows),
            "errors": errors,
            "mixed_strategy_meshes": mixed,
            "meshes_with_boundary_joints": with_joints,
        },
        "flag_counts": dict(sorted(total_flags.items())),
        "strategy_region_use": dict(sorted(strategy_use.items())),
        "strategy_part_use": dict(sorted(strategy_part_use.items())),
        "strategy_volume_use_mm3": {
            strategy_id: round(value, 3)
            for strategy_id, value in sorted(strategy_volume_use.items())
        },
        "strategy_area_use_mm2": {
            strategy_id: round(value, 3)
            for strategy_id, value in sorted(strategy_area_use.items())
        },
        "rows": rows,
        "artifacts_dir": str(eval_dir),
    }


def _merge_int_mapping(target: dict[str, int], value: object) -> None:
    if not isinstance(value, dict):
        return
    for key, raw in value.items():
        try:
            count = int(raw)
        except (TypeError, ValueError):
            continue
        target[str(key)] = target.get(str(key), 0) + count


def _merge_float_mapping(target: dict[str, float], value: object) -> None:
    if not isinstance(value, dict):
        return
    for key, raw in value.items():
        try:
            amount = float(raw)
        except (TypeError, ValueError):
            continue
        target[str(key)] = target.get(str(key), 0.0) + amount


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "mesh",
        "status",
        "overall_score",
        "regions",
        "assignments",
        "parts",
        "parts_per_region",
        "joints",
        "strategy_diversity",
        "strategy_mix",
        "strategy_part_burden",
        "strategy_volume_burden_mm3",
        "strategy_area_burden_mm2",
        "dominant_part_strategy",
        "source_part_reuse_count",
        "source_part_shared_count",
        "source_winner",
        "warning_count",
        "flags",
        "elapsed_s",
        "hybrid_plan",
        "source_ranking",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return repr(value)
    return value


def _summary_markdown(
    evaluation: dict[str, object], eval_json: Path, eval_csv: Path, report_html: Path
) -> str:
    counts = evaluation["counts"]
    assert isinstance(counts, dict)
    return "\n".join(
        [
            "# Hybrid Benchmark Evaluation",
            "",
            f"- Status: **{str(evaluation['status']).upper()}**",
            f"- Meshes: {counts['meshes']}",
            f"- Errors: {counts['errors']}",
            f"- Mixed strategy meshes: {counts['mixed_strategy_meshes']}",
            f"- Meshes with boundary joints: {counts['meshes_with_boundary_joints']}",
            f"- Evaluation JSON: `{eval_json}`",
            f"- Evaluation CSV: `{eval_csv}`",
            f"- Report HTML: `{report_html}`",
            "",
        ]
    )


def _print_row(row: dict[str, object]) -> None:
    print(
        f"{row['mesh']}: {str(row['status']).upper()} "
        f"score={float(row['overall_score']):.3f} "
        f"regions={row['regions']} parts={row['parts']} joints={row['joints']} "
        f"mix={row['strategy_mix']} flags={row['flags']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
