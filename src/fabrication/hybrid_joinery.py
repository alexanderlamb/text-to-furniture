"""Heuristic AABB joinery helpers for hybrid fabrication plans."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

from fabrication.contracts import (
    HybridFabricationPlan,
    HybridRegion,
    Joint,
    Part,
    RegionStrategyAssignment,
    Vec3,
)

DEFAULT_BOUNDARY_DISTANCE_MM = 3.175
EPSILON_MM = 1.0e-6
AXES = ("x", "y", "z")


@dataclass(frozen=True)
class AabbBoundary:
    """Classified relationship between two axis-aligned bounding boxes."""

    boundary_class: str
    distance_mm: float
    gap_dimensions_mm: Vec3
    overlap_dimensions_mm: Vec3
    separation_axes: tuple[str, ...]
    overlap_axes: tuple[str, ...]


def classify_aabb_boundary(
    left_min: Vec3,
    left_max: Vec3,
    right_min: Vec3,
    right_max: Vec3,
    *,
    max_boundary_distance_mm: float = DEFAULT_BOUNDARY_DISTANCE_MM,
) -> AabbBoundary | None:
    """Return a boundary classification when two AABBs touch or sit nearby.

    The classifier is intentionally geometric and conservative: it uses only
    AABB interval gaps/overlaps and treats anything beyond the supplied
    tolerance as non-adjacent.
    """

    max_distance = float(max_boundary_distance_mm)
    if max_distance < 0.0:
        raise ValueError("max_boundary_distance_mm must be non-negative")

    left_lo, left_hi = _normalize_aabb(left_min, left_max)
    right_lo, right_hi = _normalize_aabb(right_min, right_max)

    gaps: list[float] = []
    overlaps: list[float] = []
    for axis_index in range(3):
        gap = _interval_gap(
            left_lo[axis_index],
            left_hi[axis_index],
            right_lo[axis_index],
            right_hi[axis_index],
        )
        overlap = _interval_overlap(
            left_lo[axis_index],
            left_hi[axis_index],
            right_lo[axis_index],
            right_hi[axis_index],
        )
        gaps.append(gap)
        overlaps.append(overlap)

    distance = math.sqrt(sum(gap * gap for gap in gaps))
    if distance > max_distance + EPSILON_MM:
        return None

    positive_gap_axes = tuple(axis for axis, gap in zip(AXES, gaps) if gap > EPSILON_MM)
    overlap_axes = tuple(
        axis for axis, overlap in zip(AXES, overlaps) if overlap > EPSILON_MM
    )
    touching_axes = tuple(
        axis
        for axis, gap, overlap in zip(AXES, gaps, overlaps)
        if gap <= EPSILON_MM and overlap <= EPSILON_MM
    )

    if distance <= EPSILON_MM:
        if len(overlap_axes) == 3:
            boundary_class = "overlap"
            separation_axes = ()
        elif len(touching_axes) == 1 and len(overlap_axes) == 2:
            boundary_class = "touching_face"
            separation_axes = touching_axes
        elif len(touching_axes) == 2 and len(overlap_axes) == 1:
            boundary_class = "touching_edge"
            separation_axes = touching_axes
        elif len(touching_axes) == 3:
            boundary_class = "touching_corner"
            separation_axes = touching_axes
        else:
            boundary_class = "touching"
            separation_axes = touching_axes
    else:
        if len(positive_gap_axes) == 1 and len(overlap_axes) == 2:
            boundary_class = "near_face"
        elif len(positive_gap_axes) == 2 and len(overlap_axes) == 1:
            boundary_class = "near_edge"
        elif len(positive_gap_axes) == 3:
            boundary_class = "near_corner"
        else:
            boundary_class = "nearby"
        separation_axes = positive_gap_axes

    return AabbBoundary(
        boundary_class=boundary_class,
        distance_mm=_round_mm(distance),
        gap_dimensions_mm=_round_vec3(gaps),
        overlap_dimensions_mm=_round_vec3(overlaps),
        separation_axes=separation_axes,
        overlap_axes=overlap_axes,
    )


def synthesize_hybrid_boundary_joints(
    regions: Sequence[HybridRegion],
    assignments: Sequence[RegionStrategyAssignment],
    parts: Sequence[Part],
    *,
    max_boundary_distance_mm: float = DEFAULT_BOUNDARY_DISTANCE_MM,
    strategy_id: str = "hybrid_composition",
) -> list[Joint]:
    """Create deterministic hybrid boundary joints between assigned regions."""

    max_distance = float(max_boundary_distance_mm)
    if max_distance < 0.0:
        raise ValueError("max_boundary_distance_mm must be non-negative")

    region_by_id = {region.region_id: region for region in regions}
    known_part_ids = {part.part_id for part in parts}
    sorted_assignments = sorted(
        assignments,
        key=lambda assignment: (
            assignment.region_id,
            assignment.assignment_id,
            assignment.strategy_id,
        ),
    )

    joints: list[Joint] = []
    for left_index, left_assignment in enumerate(sorted_assignments):
        left_region = region_by_id.get(left_assignment.region_id)
        if left_region is None:
            continue
        left_part_ids = _resolved_part_ids(left_assignment, known_part_ids)
        if not left_part_ids:
            continue

        for right_assignment in sorted_assignments[left_index + 1 :]:
            if right_assignment.region_id == left_assignment.region_id:
                continue

            right_region = region_by_id.get(right_assignment.region_id)
            if right_region is None:
                continue
            right_part_ids = _resolved_part_ids(right_assignment, known_part_ids)
            if not right_part_ids:
                continue

            boundary = classify_aabb_boundary(
                left_region.aabb_min,
                left_region.aabb_max,
                right_region.aabb_min,
                right_region.aabb_max,
                max_boundary_distance_mm=max_distance,
            )
            if boundary is None:
                continue

            joints.append(
                _make_joint(
                    left_region,
                    right_region,
                    left_assignment,
                    right_assignment,
                    left_part_ids,
                    right_part_ids,
                    boundary,
                    max_distance,
                    strategy_id,
                )
            )

    return sorted(
        _dedupe_joint_ids(joints),
        key=lambda joint: (
            str(joint.metadata["region_pair_key"]),
            str(joint.metadata["assignment_pair_key"]),
            joint.joint_id,
        ),
    )


def add_hybrid_boundary_joints(
    plan: HybridFabricationPlan,
    *,
    max_boundary_distance_mm: float = DEFAULT_BOUNDARY_DISTANCE_MM,
    strategy_id: str = "hybrid_composition",
) -> HybridFabricationPlan:
    """Append synthesized hybrid boundary joints to a plan, avoiding duplicates."""

    generated = synthesize_hybrid_boundary_joints(
        plan.regions,
        plan.assignments,
        plan.parts,
        max_boundary_distance_mm=max_boundary_distance_mm,
        strategy_id=strategy_id,
    )
    existing_ids = {joint.joint_id for joint in plan.joints}
    plan.joints.extend(
        joint for joint in generated if joint.joint_id not in existing_ids
    )
    return plan


synthesize_hybrid_joints = synthesize_hybrid_boundary_joints


def _make_joint(
    left_region: HybridRegion,
    right_region: HybridRegion,
    left_assignment: RegionStrategyAssignment,
    right_assignment: RegionStrategyAssignment,
    left_part_ids: list[str],
    right_part_ids: list[str],
    boundary: AabbBoundary,
    max_boundary_distance_mm: float,
    strategy_id: str,
) -> Joint:
    strategy_pair = [left_assignment.strategy_id, right_assignment.strategy_id]
    region_pair = [left_region.region_id, right_region.region_id]
    assignment_pair = [
        left_assignment.assignment_id,
        right_assignment.assignment_id,
    ]
    part_ids = sorted(set(left_part_ids + right_part_ids))
    metadata = {
        "generated_by": "fabrication.hybrid_joinery",
        "boundary_class": boundary.boundary_class,
        "distance_mm": boundary.distance_mm,
        "max_boundary_distance_mm": _round_mm(max_boundary_distance_mm),
        "strategy_pair": strategy_pair,
        "strategy_pair_key": "::".join(strategy_pair),
        "region_ids": region_pair,
        "region_pair_key": "::".join(region_pair),
        "assignment_ids": assignment_pair,
        "assignment_pair_key": "::".join(assignment_pair),
        "left_region_id": left_region.region_id,
        "right_region_id": right_region.region_id,
        "left_part_ids": list(left_part_ids),
        "right_part_ids": list(right_part_ids),
        "gap_dimensions_mm": list(boundary.gap_dimensions_mm),
        "overlap_dimensions_mm": list(boundary.overlap_dimensions_mm),
        "separation_axes": list(boundary.separation_axes),
        "overlap_axes": list(boundary.overlap_axes),
        "joinery_hint": _joinery_hint(strategy_pair, boundary.boundary_class),
    }
    return Joint(
        joint_id=(
            f"hybrid_boundary_"
            f"{_slug(left_assignment.assignment_id)}__"
            f"{_slug(right_assignment.assignment_id)}"
        ),
        strategy_id=strategy_id,
        part_ids=part_ids,
        kind=_joint_kind(boundary.boundary_class),
        metadata=metadata,
    )


def _joint_kind(boundary_class: str) -> str:
    if "face" in boundary_class:
        return "hybrid_face_boundary"
    if "edge" in boundary_class:
        return "hybrid_edge_boundary"
    if "corner" in boundary_class:
        return "hybrid_corner_boundary"
    if boundary_class == "overlap":
        return "hybrid_overlap_boundary"
    return "hybrid_near_boundary"


def _joinery_hint(strategy_pair: Sequence[str], boundary_class: str) -> str:
    strategies = set(strategy_pair)
    if boundary_class == "overlap":
        return "resolve_overlap_before_joinery"
    if boundary_class.startswith("near_") or boundary_class == "nearby":
        return "bridge_gap_with_splice_or_fastener"
    if len(strategies) == 1:
        return "strategy_native_boundary"
    if {"planar_skin", "voxel_blocks"}.issubset(strategies):
        return "panel_to_block_fastener"
    if {"planar_skin", "contour_stack"}.issubset(strategies):
        return "panel_to_laminate_tab_or_rabbet"
    if {"contour_stack", "voxel_blocks"}.issubset(strategies):
        return "laminate_to_block_dowel_or_screw"
    return "hybrid_mechanical_fastener"


def _resolved_part_ids(
    assignment: RegionStrategyAssignment, known_part_ids: set[str]
) -> list[str]:
    unique_part_ids = set(assignment.part_ids)
    if known_part_ids:
        unique_part_ids &= known_part_ids
    return sorted(unique_part_ids)


def _dedupe_joint_ids(joints: Sequence[Joint]) -> list[Joint]:
    seen: dict[str, int] = {}
    deduped: list[Joint] = []
    for joint in joints:
        count = seen.get(joint.joint_id, 0)
        seen[joint.joint_id] = count + 1
        if count:
            joint.joint_id = f"{joint.joint_id}_{count + 1:02d}"
        deduped.append(joint)
    return deduped


def _normalize_aabb(aabb_min: Vec3, aabb_max: Vec3) -> tuple[Vec3, Vec3]:
    lows = tuple(float(min(low, high)) for low, high in zip(aabb_min, aabb_max))
    highs = tuple(float(max(low, high)) for low, high in zip(aabb_min, aabb_max))
    return lows, highs


def _interval_gap(
    left_min: float, left_max: float, right_min: float, right_max: float
) -> float:
    if left_max < right_min:
        return float(right_min - left_max)
    if right_max < left_min:
        return float(left_min - right_max)
    return 0.0


def _interval_overlap(
    left_min: float, left_max: float, right_min: float, right_max: float
) -> float:
    return max(0.0, float(min(left_max, right_max) - max(left_min, right_min)))


def _round_vec3(values: Sequence[float]) -> Vec3:
    return tuple(_round_mm(value) for value in values)


def _round_mm(value: float) -> float:
    return round(float(value), 6)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "assignment"
