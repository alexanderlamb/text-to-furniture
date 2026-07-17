"""Hybrid fabrication compositor.

This layer sits above the strategy tournament: it runs every generator, carves
the mesh into coarse spatial regions, assigns each region to the strategy output
that best fits it, then emits one composed plan.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from fabrication.context import FabricationContext, build_fabrication_context
from fabrication.contracts import (
    FabricationConfig,
    FabricationPlan,
    HybridFabricationPlan,
    HybridRegion,
    Joint,
    Operation,
    Part,
    RegionStrategyAssignment,
    Vec3,
)
from fabrication.scoring import clamp01, score_from_components
from fabrication.tournament import run_tournament, write_tournament_artifacts


def run_hybrid_composition(
    config: FabricationConfig,
    artifacts_dir: Path | None = None,
    strategy_ids: Iterable[str] | None = None,
    max_regions: int | None = None,
) -> Dict[str, object]:
    """Run strategy generators and compose a prototype hybrid plan."""

    context = build_fabrication_context(config)
    tournament_dir = (
        Path(artifacts_dir) / "source_strategies" if artifacts_dir else None
    )
    tournament = run_tournament(
        config,
        artifacts_dir=tournament_dir,
        strategy_ids=strategy_ids or config.strategies,
    )
    plans = tournament["plans"]
    assert isinstance(plans, dict)

    regions = _regionize(
        context=context,
        max_regions=max_regions or int(config.max_hybrid_regions),
    )
    hybrid_plan = compose_hybrid_plan(
        context=context,
        regions=regions,
        source_plans=plans,
    )

    return {
        "schema_version": "fabrication.hybrid.v0",
        "context": context.summary_payload(),
        "hybrid_plan": hybrid_plan,
        "source_tournament": tournament,
    }


def compose_hybrid_plan(
    *,
    context: FabricationContext,
    regions: Sequence[HybridRegion],
    source_plans: Dict[str, FabricationPlan],
) -> HybridFabricationPlan:
    viable_plans = {
        strategy_id: plan
        for strategy_id, plan in source_plans.items()
        if plan.status != "error" and plan.parts
    }
    warnings: List[str] = []
    if not viable_plans:
        return HybridFabricationPlan(
            status="error",
            regions=list(regions),
            warnings=["No viable source strategy plans were available."],
            scores={"overall": 0.0},
        )

    assignments: List[RegionStrategyAssignment] = []
    composed_parts: List[Part] = []
    composed_part_by_id: dict[str, Part] = {}
    composed_part_id_by_source: dict[tuple[str, str], str] = {}
    operations: List[Operation] = []
    source_part_usage: dict[tuple[str, str], int] = {}

    for region in regions:
        selected = _select_region_strategy(
            region,
            viable_plans,
            source_part_usage=source_part_usage,
        )
        if selected is None:
            warnings.append(f"No strategy parts overlapped region {region.region_id}.")
            continue

        strategy_id, source_parts, fit_score, reasons, metadata = selected
        assignment = RegionStrategyAssignment(
            assignment_id=f"assign_{len(assignments):03d}",
            region_id=region.region_id,
            strategy_id=strategy_id,
            part_ids=[],
            fit_score=fit_score,
            reason_codes=reasons,
            metadata=metadata,
        )

        for source_part in source_parts:
            source_key = _source_part_key(source_part)
            existing_part_id = composed_part_id_by_source.get(source_key)
            if existing_part_id is not None:
                existing_part = composed_part_by_id[existing_part_id]
                _add_part_assignment_metadata(
                    existing_part,
                    region_id=region.region_id,
                    assignment_id=assignment.assignment_id,
                )
                assignment.part_ids.append(existing_part_id)
                source_part_usage[source_key] = source_part_usage.get(source_key, 0) + 1
                continue

            part_id = f"{assignment.assignment_id}__{source_part.part_id}"
            composed = replace(
                source_part,
                part_id=part_id,
                metadata={
                    **source_part.metadata,
                    "source_part_id": source_part.part_id,
                    "source_strategy_id": source_part.strategy_id,
                    "hybrid_region_id": region.region_id,
                    "hybrid_assignment_id": assignment.assignment_id,
                    "hybrid_region_ids": [region.region_id],
                    "hybrid_assignment_ids": [assignment.assignment_id],
                },
            )
            composed_parts.append(composed)
            composed_part_by_id[part_id] = composed
            composed_part_id_by_source[source_key] = part_id
            assignment.part_ids.append(part_id)
            source_part_usage[source_key] = source_part_usage.get(source_key, 0) + 1

        assignments.append(assignment)
        operations.append(
            Operation(
                operation_id=f"hybrid_place_{assignment.assignment_id}",
                strategy_id="hybrid_compositor",
                kind="region_strategy_assignment",
                part_ids=list(assignment.part_ids),
                metadata={
                    "region_id": region.region_id,
                    "source_strategy_id": strategy_id,
                    "fit_score": round(float(fit_score), 6),
                    "reason_codes": reasons,
                },
            )
        )

    joints = _synthesize_joints(
        regions=regions,
        assignments=assignments,
        parts=composed_parts,
    )
    scores = _score_hybrid(
        context=context,
        regions=regions,
        assignments=assignments,
        parts=composed_parts,
        joints=joints,
        warnings=warnings,
        source_plans=viable_plans,
    )
    status = (
        "ok" if assignments and not warnings else "warning" if assignments else "error"
    )
    source_part_reuse = _source_part_reuse(composed_parts)
    source_part_sharing = _source_part_sharing(composed_parts)
    return HybridFabricationPlan(
        status=status,
        regions=list(regions),
        assignments=assignments,
        parts=composed_parts,
        joints=joints,
        operations=operations,
        scores=scores,
        warnings=warnings,
        source_strategy_scores={
            strategy_id: dict(plan.scores) for strategy_id, plan in source_plans.items()
        },
        debug={
            "source_strategy_count": len(source_plans),
            "viable_strategy_count": len(viable_plans),
            "strategy_mix": _strategy_mix(assignments),
            "source_part_reuse_count": len(source_part_reuse),
            "source_part_reuse": source_part_reuse,
            "source_part_shared_count": len(source_part_sharing),
            "source_part_sharing": source_part_sharing,
            "region_assignment_debug": [
                {
                    "region_id": assignment.region_id,
                    "strategy_id": assignment.strategy_id,
                    "part_count": len(assignment.part_ids),
                    "fit_score": round(float(assignment.fit_score), 6),
                    "reason_codes": list(assignment.reason_codes),
                }
                for assignment in assignments
            ],
        },
    )


def write_hybrid_artifacts(result: Dict[str, object], artifacts_dir: Path) -> None:
    hybrid_dir = Path(artifacts_dir)
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    plan = result["hybrid_plan"]
    assert isinstance(plan, HybridFabricationPlan)
    _write_json(hybrid_dir / "hybrid_plan.json", plan.to_payload())

    source_tournament = result.get("source_tournament")
    if isinstance(source_tournament, dict):
        source_dir = hybrid_dir / "source_strategies"
        write_tournament_artifacts(source_tournament, source_dir)

    summary = {
        key: value
        for key, value in result.items()
        if key not in {"hybrid_plan", "source_tournament"}
    }
    summary["hybrid"] = {
        "status": plan.status,
        "overall_score": float(plan.scores.get("overall", 0.0)),
        "region_count": len(plan.regions),
        "assignment_count": len(plan.assignments),
        "part_count": len(plan.parts),
        "joint_count": len(plan.joints),
        "strategy_mix": plan.debug.get("strategy_mix", {}),
    }
    source_tournament = result.get("source_tournament")
    if isinstance(source_tournament, dict):
        summary["source_ranking"] = source_tournament.get("ranking", [])
    _write_json(hybrid_dir / "hybrid_summary.json", summary)


def _regionize(context: FabricationContext, max_regions: int) -> List[HybridRegion]:
    try:
        from fabrication.regioning import regionize_mesh

        return list(regionize_mesh(context, max_regions=max_regions))
    except ImportError:
        return _fallback_regionize(context, max_regions=max_regions)


def _fallback_regionize(
    context: FabricationContext, max_regions: int
) -> List[HybridRegion]:
    bounds = np.asarray(context.mesh.bounds, dtype=float)
    extents = bounds[1] - bounds[0]
    split_axis = int(np.argmax(extents))
    region_count = max(1, min(int(max_regions), 3))
    cuts = np.linspace(bounds[0][split_axis], bounds[1][split_axis], region_count + 1)
    regions: List[HybridRegion] = []

    for index in range(region_count):
        aabb_min = bounds[0].copy()
        aabb_max = bounds[1].copy()
        aabb_min[split_axis] = cuts[index]
        aabb_max[split_axis] = cuts[index + 1]
        dims = np.maximum(aabb_max - aabb_min, 0.0)
        kind = _classify_region(index=index, count=region_count, dims=dims)
        regions.append(
            HybridRegion(
                region_id=f"region_{index:03d}",
                kind=kind,
                aabb_min=_vec3(aabb_min),
                aabb_max=_vec3(aabb_max),
                volume_mm3=float(np.prod(dims)),
                surface_area_mm2=float(
                    2.0 * (dims[0] * dims[1] + dims[0] * dims[2] + dims[1] * dims[2])
                ),
                metadata={
                    "regioning_method": "fallback_longest_axis_bands",
                    "split_axis": ["x", "y", "z"][split_axis],
                    "band_index": index,
                    "band_count": region_count,
                    "dimensions_mm": [float(v) for v in dims],
                },
            )
        )
    return regions


def _classify_region(index: int, count: int, dims: np.ndarray) -> str:
    ratios = np.sort(dims) / max(float(np.max(dims)), 1e-9)
    if ratios[0] < 0.20:
        return "flat_band"
    if count > 1 and index in {0, count - 1}:
        return "shell_band"
    if ratios[1] > 0.55:
        return "blocky_band"
    return "layer_band"


def _select_region_strategy(
    region: HybridRegion,
    plans: Dict[str, FabricationPlan],
    *,
    source_part_usage: Mapping[tuple[str, str], int] | None = None,
) -> tuple[str, List[Part], float, List[str], Dict[str, object]] | None:
    candidates = []
    region_role = _assignment_role(region)
    usage = source_part_usage or {}
    for strategy_id, plan in plans.items():
        scored_parts = _parts_for_region(region, plan.parts)
        if not scored_parts:
            continue
        selected_scored_parts = _select_parts_for_assignment(
            scored_parts=scored_parts,
            region_role=region_role,
            strategy_id=strategy_id,
            source_part_usage=usage,
        )
        selected_parts = [part for part, _score in selected_scored_parts]
        coverage = sum(score for _part, score in selected_scored_parts)
        coverage = coverage / max(float(len(selected_parts)), 1.0)
        affinity = _strategy_affinity(region_role, strategy_id)
        role_penalty = _role_mismatch_penalty(region_role, strategy_id)
        candidate_reused_count = sum(
            1 for part, _score in scored_parts if _source_part_use_count(part, usage)
        )
        selected_reused_count = sum(
            1
            for part, _score in selected_scored_parts
            if _source_part_use_count(part, usage)
        )
        fit_score = clamp01(
            0.35 * float(plan.scores.get("overall", 0.0))
            + 0.25 * coverage
            + 0.40 * affinity
            - role_penalty
        )
        reasons = [
            f"region_kind:{region.kind}",
            f"assignment_role:{region_role}",
            f"source_strategy:{strategy_id}",
            f"overlapping_parts:{len(scored_parts)}",
        ]
        if affinity >= 0.80:
            reasons.append("strategy_affinity")
        if selected_reused_count:
            reasons.append("source_part_reuse_penalty")
        candidates.append(
            (
                fit_score,
                strategy_id,
                selected_parts,
                reasons,
                {
                    "coverage_score": round(float(coverage), 6),
                    "strategy_affinity": round(float(affinity), 6),
                    "role_mismatch_penalty": round(float(role_penalty), 6),
                    "source_overall_score": float(plan.scores.get("overall", 0.0)),
                    "candidate_part_count": len(scored_parts),
                    "selected_part_count": len(selected_parts),
                    "candidate_reused_part_count": candidate_reused_count,
                    "selected_reused_part_count": selected_reused_count,
                    "selection_cap": _assignment_part_cap(region_role, strategy_id),
                    "selection_min": _assignment_part_min(region_role, strategy_id),
                },
            )
        )

    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            -round(float(item[0]), 12),
            _strategy_tiebreak(item[1], region_role),
            item[1],
        )
    )
    fit_score, strategy_id, selected_parts, reasons, metadata = candidates[0]
    return strategy_id, selected_parts, fit_score, reasons, metadata


def _parts_for_region(
    region: HybridRegion, parts: Sequence[Part]
) -> List[tuple[Part, float]]:
    scored: List[tuple[Part, float]] = []
    for part in parts:
        intersection = _aabb_intersection_volume(
            region.aabb_min, region.aabb_max, part.aabb_min, part.aabb_max
        )
        part_volume = max(_aabb_volume(part.aabb_min, part.aabb_max), 1.0)
        region_volume = max(_aabb_volume(region.aabb_min, region.aabb_max), 1.0)
        center_bonus = (
            0.25
            if _aabb_contains(
                region.aabb_min,
                region.aabb_max,
                _aabb_center(part.aabb_min, part.aabb_max),
            )
            else 0.0
        )
        score = clamp01(
            0.65 * min(1.0, intersection / part_volume)
            + 0.25 * min(1.0, intersection / region_volume)
            + center_bonus
        )
        if score > 0.0:
            scored.append((part, score))
    scored.sort(key=lambda item: (-round(float(item[1]), 12), item[0].part_id))
    return scored


def _select_parts_for_assignment(
    *,
    scored_parts: Sequence[tuple[Part, float]],
    region_role: str,
    strategy_id: str,
    source_part_usage: Mapping[tuple[str, str], int] | None = None,
) -> List[tuple[Part, float]]:
    cap = _assignment_part_cap(region_role, strategy_id)
    if not scored_parts:
        return []
    usage = source_part_usage or {}
    adjusted_parts = []
    for part, score in scored_parts:
        use_count = _source_part_use_count(part, usage)
        reuse_factor = 0.32**use_count if use_count else 1.0
        adjusted_parts.append(
            (part, float(score) * reuse_factor, float(score), use_count)
        )
    adjusted_parts.sort(
        key=lambda item: (
            -round(float(item[1]), 12),
            -round(float(item[2]), 12),
            int(item[3]),
            item[0].part_id,
        )
    )
    best_score = float(adjusted_parts[0][1])
    threshold = max(0.08, best_score * 0.42)
    filtered = [item for item in adjusted_parts if float(item[1]) >= threshold]
    minimum = min(
        _assignment_part_min(region_role, strategy_id), len(scored_parts), cap
    )
    if len(filtered) < minimum:
        filtered = list(adjusted_parts[:minimum])
    return [
        (part, adjusted_score) for part, adjusted_score, _score, _use in filtered[:cap]
    ]


def _assignment_part_cap(region_role: str, strategy_id: str) -> int:
    if strategy_id == "waffle_ribs":
        return 4 if region_role == "rib_band" else 3
    if strategy_id == "planar_skin":
        return 6 if region_role in {"shell_band", "flat_band"} else 4
    if strategy_id == "contour_stack":
        return 5 if region_role in {"layer_band", "rib_band"} else 5
    if strategy_id == "voxel_blocks":
        return 4
    return 6


def _assignment_part_min(region_role: str, strategy_id: str) -> int:
    if strategy_id == "planar_skin" and region_role in {"shell_band", "flat_band"}:
        return 2
    if strategy_id == "waffle_ribs" and region_role == "rib_band":
        return 2
    return 1


def _strategy_affinity(region_kind: str, strategy_id: str) -> float:
    table = {
        "flat_band": {
            "planar_skin": 1.0,
            "contour_stack": 0.45,
            "waffle_ribs": 0.35,
            "voxel_blocks": 0.35,
        },
        "shell_band": {
            "planar_skin": 0.75,
            "contour_stack": 0.85,
            "waffle_ribs": 0.45,
            "voxel_blocks": 0.45,
        },
        "layer_band": {
            "planar_skin": 0.45,
            "contour_stack": 1.0,
            "waffle_ribs": 0.75,
            "voxel_blocks": 0.60,
        },
        "rib_band": {
            "planar_skin": 0.35,
            "contour_stack": 0.80,
            "waffle_ribs": 1.0,
            "voxel_blocks": 0.55,
        },
        "blocky_band": {
            "planar_skin": 0.50,
            "contour_stack": 0.50,
            "waffle_ribs": 0.60,
            "voxel_blocks": 1.0,
        },
    }
    return float(table.get(region_kind, {}).get(strategy_id, 0.50))


def _assignment_role(region: HybridRegion) -> str:
    """Interpret coarse region metadata as a fabrication role."""

    if region.kind == "shell_band" and _is_interior_band(region):
        if _is_structural_rib_candidate(region):
            return "rib_band"
        return "layer_band"

    if region.kind != "blocky_band":
        return region.kind

    if _is_outer_band(region):
        return "shell_band"
    return region.kind


def _role_mismatch_penalty(region_kind: str, strategy_id: str) -> float:
    if region_kind in {"flat_band", "shell_band"} and strategy_id == "voxel_blocks":
        return 0.22
    if region_kind == "shell_band" and strategy_id == "waffle_ribs":
        return 0.08
    if region_kind == "flat_band" and strategy_id == "waffle_ribs":
        return 0.12
    if region_kind == "blocky_band" and strategy_id == "contour_stack":
        return 0.08
    return 0.0


def _strategy_tiebreak(strategy_id: str, region_kind: str) -> int:
    order = {
        "flat_band": ["planar_skin", "contour_stack", "voxel_blocks"],
        "shell_band": ["contour_stack", "planar_skin", "voxel_blocks"],
        "layer_band": ["contour_stack", "voxel_blocks", "planar_skin"],
        "rib_band": ["waffle_ribs", "contour_stack", "voxel_blocks", "planar_skin"],
        "blocky_band": ["voxel_blocks", "planar_skin", "contour_stack"],
    }.get(region_kind, [])
    return order.index(strategy_id) if strategy_id in order else 99


def _is_outer_band(region: HybridRegion) -> bool:
    band_index = region.metadata.get("band_index")
    band_count = region.metadata.get("band_count")
    try:
        band_index_int = int(band_index)
        band_count_int = int(band_count)
    except (TypeError, ValueError):
        return False
    return band_count_int >= 3 and band_index_int in {0, band_count_int - 1}


def _is_interior_band(region: HybridRegion) -> bool:
    band_index = region.metadata.get("band_index")
    band_count = region.metadata.get("band_count")
    try:
        band_index_int = int(band_index)
        band_count_int = int(band_count)
    except (TypeError, ValueError):
        return False
    return band_count_int >= 4 and 0 < band_index_int < band_count_int - 1


def _is_structural_rib_candidate(region: HybridRegion) -> bool:
    """Return true when a shell band has enough local evidence to justify ribs."""

    occupancy = region.metadata.get("occupancy_proxy", {})
    if not isinstance(occupancy, dict):
        return False

    fill_ratio = _float_metadata(occupancy.get("fill_ratio"))
    sample_inside_count = _int_metadata(occupancy.get("sample_inside_count"))
    vertex_count = _int_metadata(occupancy.get("vertex_count"))
    face_centroid_count = _int_metadata(occupancy.get("face_centroid_count"))
    shrink_ratio = _float_metadata(region.metadata.get("transverse_shrink_ratio"))

    if 0.0 < shrink_ratio < 0.20 and vertex_count == 0:
        return False

    if fill_ratio >= 0.18 or sample_inside_count >= 2:
        return True
    if vertex_count >= 8:
        return True
    return face_centroid_count >= 16


def _float_metadata(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_metadata(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _synthesize_joints(
    *,
    regions: Sequence[HybridRegion],
    assignments: Sequence[RegionStrategyAssignment],
    parts: Sequence[Part],
) -> List[Joint]:
    try:
        from fabrication.hybrid_joinery import synthesize_hybrid_joints

        return list(synthesize_hybrid_joints(regions, assignments, parts))
    except ImportError:
        return _fallback_joints(regions=regions, assignments=assignments, parts=parts)


def _fallback_joints(
    *,
    regions: Sequence[HybridRegion],
    assignments: Sequence[RegionStrategyAssignment],
    parts: Sequence[Part],
) -> List[Joint]:
    by_region = {assignment.region_id: assignment for assignment in assignments}
    joints: List[Joint] = []
    part_lookup = {part.part_id: part for part in parts}

    for index, region_a in enumerate(regions):
        assignment_a = by_region.get(region_a.region_id)
        if assignment_a is None:
            continue
        for region_b in regions[index + 1 :]:
            assignment_b = by_region.get(region_b.region_id)
            if assignment_b is None:
                continue
            distance = _aabb_distance(
                region_a.aabb_min,
                region_a.aabb_max,
                region_b.aabb_min,
                region_b.aabb_max,
            )
            if distance > 1e-6:
                continue
            part_a = _representative_part(assignment_a.part_ids, part_lookup)
            part_b = _representative_part(assignment_b.part_ids, part_lookup)
            if part_a is None or part_b is None:
                continue
            joints.append(
                Joint(
                    joint_id=f"hybrid_boundary_{len(joints):03d}",
                    strategy_id="hybrid_compositor",
                    part_ids=[part_a.part_id, part_b.part_id],
                    kind="strategy_boundary_joint",
                    metadata={
                        "region_a": region_a.region_id,
                        "region_b": region_b.region_id,
                        "assignment_a": assignment_a.assignment_id,
                        "assignment_b": assignment_b.assignment_id,
                        "strategy_pair": [
                            assignment_a.strategy_id,
                            assignment_b.strategy_id,
                        ],
                        "boundary_distance_mm": float(distance),
                        "method": "fallback_touching_region_aabb",
                    },
                )
            )
    return joints


def _representative_part(
    part_ids: Sequence[str], part_lookup: Dict[str, Part]
) -> Part | None:
    available = [part_lookup[part_id] for part_id in part_ids if part_id in part_lookup]
    if not available:
        return None
    return max(available, key=lambda part: (float(part.volume_mm3), part.part_id))


def _score_hybrid(
    *,
    context: FabricationContext,
    regions: Sequence[HybridRegion],
    assignments: Sequence[RegionStrategyAssignment],
    parts: Sequence[Part],
    joints: Sequence[Joint],
    warnings: Sequence[str],
    source_plans: Dict[str, FabricationPlan],
) -> Dict[str, float]:
    assigned_region_ratio = len(assignments) / max(float(len(regions)), 1.0)
    strategy_diversity = len({assignment.strategy_id for assignment in assignments})
    source_score = 0.0
    if assignments:
        source_score = sum(float(a.fit_score) for a in assignments) / len(assignments)
    total_part_volume = sum(float(part.volume_mm3) for part in parts)
    volume_ratio = total_part_volume / max(float(context.mesh_volume_mm3), 1.0)

    return score_from_components(
        {
            "fidelity": clamp01(
                0.25 + 0.45 * assigned_region_ratio + 0.30 * source_score
            ),
            "material_efficiency": clamp01(1.0 / (1.0 + max(0.0, volume_ratio - 1.0))),
            "assembly_simplicity": clamp01(
                1.0 / (1.0 + len(parts) / 36.0 + len(joints) / 18.0)
            ),
            "strength_proxy": clamp01(
                0.45 + 0.15 * len(joints) + 0.10 * min(strategy_diversity, 3)
            ),
            "part_count": clamp01(1.0 / (1.0 + len(parts) / 48.0)),
            "risk": 0.70 if not warnings else 0.50,
        }
    )


def _strategy_mix(assignments: Sequence[RegionStrategyAssignment]) -> Dict[str, int]:
    mix: Dict[str, int] = {}
    for assignment in assignments:
        mix[assignment.strategy_id] = mix.get(assignment.strategy_id, 0) + 1
    return dict(sorted(mix.items()))


def _source_part_key(part: Part) -> tuple[str, str]:
    return (str(part.strategy_id), str(part.part_id))


def _source_part_use_count(
    part: Part, source_part_usage: Mapping[tuple[str, str], int]
) -> int:
    return int(source_part_usage.get(_source_part_key(part), 0))


def _add_part_assignment_metadata(
    part: Part, *, region_id: str, assignment_id: str
) -> None:
    region_ids = _metadata_string_list(part.metadata.get("hybrid_region_ids"))
    assignment_ids = _metadata_string_list(part.metadata.get("hybrid_assignment_ids"))
    if region_id not in region_ids:
        region_ids.append(region_id)
    if assignment_id not in assignment_ids:
        assignment_ids.append(assignment_id)
    part.metadata["hybrid_region_ids"] = region_ids
    part.metadata["hybrid_assignment_ids"] = assignment_ids


def _metadata_string_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _source_part_reuse(parts: Sequence[Part]) -> list[dict[str, object]]:
    usage: dict[tuple[str, str], list[str]] = {}
    for part in parts:
        strategy_id = str(
            part.metadata.get("source_strategy_id") or part.strategy_id or "unknown"
        )
        source_part_id = str(part.metadata.get("source_part_id") or part.part_id)
        key = (strategy_id, source_part_id)
        usage.setdefault(key, []).append(part.part_id)

    reused = []
    for (strategy_id, source_part_id), part_ids in sorted(usage.items()):
        if len(part_ids) <= 1:
            continue
        reused.append(
            {
                "source_strategy_id": strategy_id,
                "source_part_id": source_part_id,
                "use_count": len(part_ids),
                "hybrid_part_ids": list(part_ids),
            }
        )
    return reused


def _source_part_sharing(parts: Sequence[Part]) -> list[dict[str, object]]:
    shared = []
    for part in parts:
        region_ids = _metadata_string_list(part.metadata.get("hybrid_region_ids"))
        assignment_ids = _metadata_string_list(
            part.metadata.get("hybrid_assignment_ids")
        )
        if len(region_ids) <= 1 and len(assignment_ids) <= 1:
            continue
        shared.append(
            {
                "source_strategy_id": str(
                    part.metadata.get("source_strategy_id")
                    or part.strategy_id
                    or "unknown"
                ),
                "source_part_id": str(
                    part.metadata.get("source_part_id") or part.part_id
                ),
                "hybrid_part_id": part.part_id,
                "region_ids": region_ids,
                "assignment_ids": assignment_ids,
            }
        )
    return sorted(
        shared,
        key=lambda item: (
            str(item["source_strategy_id"]),
            str(item["source_part_id"]),
            str(item["hybrid_part_id"]),
        ),
    )


def _aabb_intersection_volume(
    a_min: Vec3, a_max: Vec3, b_min: Vec3, b_max: Vec3
) -> float:
    lo = np.maximum(np.asarray(a_min, dtype=float), np.asarray(b_min, dtype=float))
    hi = np.minimum(np.asarray(a_max, dtype=float), np.asarray(b_max, dtype=float))
    dims = np.maximum(hi - lo, 0.0)
    return float(np.prod(dims))


def _aabb_volume(aabb_min: Vec3, aabb_max: Vec3) -> float:
    dims = np.maximum(
        np.asarray(aabb_max, dtype=float) - np.asarray(aabb_min, dtype=float), 0.0
    )
    return float(np.prod(dims))


def _aabb_center(aabb_min: Vec3, aabb_max: Vec3) -> Vec3:
    center = (
        np.asarray(aabb_min, dtype=float) + np.asarray(aabb_max, dtype=float)
    ) * 0.5
    return _vec3(center)


def _aabb_contains(aabb_min: Vec3, aabb_max: Vec3, point: Vec3) -> bool:
    lo = np.asarray(aabb_min, dtype=float) - 1e-6
    hi = np.asarray(aabb_max, dtype=float) + 1e-6
    p = np.asarray(point, dtype=float)
    return bool(np.all(p >= lo) and np.all(p <= hi))


def _aabb_distance(a_min: Vec3, a_max: Vec3, b_min: Vec3, b_max: Vec3) -> float:
    a0 = np.asarray(a_min, dtype=float)
    a1 = np.asarray(a_max, dtype=float)
    b0 = np.asarray(b_min, dtype=float)
    b1 = np.asarray(b_max, dtype=float)
    gap = np.maximum(np.maximum(a0 - b1, b0 - a1), 0.0)
    return float(np.linalg.norm(gap))


def _vec3(values) -> Vec3:
    arr = np.asarray(values, dtype=float)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
