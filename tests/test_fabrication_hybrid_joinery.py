from __future__ import annotations

import pytest

from fabrication.contracts import (
    HybridFabricationPlan,
    HybridRegion,
    Part,
    RegionStrategyAssignment,
)
from fabrication.hybrid_joinery import (
    add_hybrid_boundary_joints,
    classify_aabb_boundary,
    synthesize_hybrid_boundary_joints,
    synthesize_hybrid_joints,
)


def _region(
    region_id: str,
    aabb_min: tuple[float, float, float],
    aabb_max: tuple[float, float, float],
) -> HybridRegion:
    return HybridRegion(
        region_id=region_id,
        kind="test_region",
        aabb_min=aabb_min,
        aabb_max=aabb_max,
    )


def _assignment(
    assignment_id: str,
    region_id: str,
    strategy_id: str,
    part_ids: list[str],
) -> RegionStrategyAssignment:
    return RegionStrategyAssignment(
        assignment_id=assignment_id,
        region_id=region_id,
        strategy_id=strategy_id,
        part_ids=part_ids,
        fit_score=1.0,
    )


def _part(part_id: str, strategy_id: str) -> Part:
    return Part(
        part_id=part_id,
        strategy_id=strategy_id,
        kind="test_part",
    )


def _signature(joints):
    return [
        (
            joint.joint_id,
            joint.kind,
            joint.part_ids,
            joint.metadata["boundary_class"],
            joint.metadata["distance_mm"],
            joint.metadata["strategy_pair"],
            joint.metadata["region_ids"],
            joint.metadata["separation_axes"],
        )
        for joint in joints
    ]


def test_classify_aabb_boundary_distinguishes_touching_and_near_faces():
    touching = classify_aabb_boundary(
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 10.0),
        (10.0, 0.0, 0.0),
        (20.0, 10.0, 10.0),
    )
    near = classify_aabb_boundary(
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 10.0),
        (12.0, 0.0, 0.0),
        (22.0, 10.0, 10.0),
        max_boundary_distance_mm=2.5,
    )

    assert touching is not None
    assert touching.boundary_class == "touching_face"
    assert touching.distance_mm == pytest.approx(0.0)
    assert touching.separation_axes == ("x",)
    assert touching.overlap_axes == ("y", "z")

    assert near is not None
    assert near.boundary_class == "near_face"
    assert near.distance_mm == pytest.approx(2.0)
    assert near.gap_dimensions_mm == (2.0, 0.0, 0.0)
    assert near.separation_axes == ("x",)


def test_synthesizes_deterministic_touching_region_joint_with_metadata():
    regions = [
        _region("region_b", (10.0, 0.0, 0.0), (20.0, 10.0, 10.0)),
        _region("region_a", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
    ]
    assignments = [
        _assignment("assign_b", "region_b", "voxel_blocks", ["part_b"]),
        _assignment("assign_a", "region_a", "planar_skin", ["part_a"]),
    ]
    parts = [_part("part_b", "voxel_blocks"), _part("part_a", "planar_skin")]

    joints = synthesize_hybrid_boundary_joints(regions, assignments, parts)
    shuffled = synthesize_hybrid_joints(
        list(reversed(regions)),
        list(reversed(assignments)),
        list(reversed(parts)),
    )

    assert _signature(joints) == _signature(shuffled)
    assert len(joints) == 1

    joint = joints[0]
    assert joint.joint_id == "hybrid_boundary_assign_a__assign_b"
    assert joint.strategy_id == "hybrid_composition"
    assert joint.kind == "hybrid_face_boundary"
    assert joint.part_ids == ["part_a", "part_b"]
    assert joint.metadata["boundary_class"] == "touching_face"
    assert joint.metadata["distance_mm"] == pytest.approx(0.0)
    assert joint.metadata["strategy_pair"] == ["planar_skin", "voxel_blocks"]
    assert joint.metadata["region_ids"] == ["region_a", "region_b"]
    assert joint.metadata["assignment_ids"] == ["assign_a", "assign_b"]
    assert joint.metadata["left_part_ids"] == ["part_a"]
    assert joint.metadata["right_part_ids"] == ["part_b"]
    assert joint.metadata["separation_axes"] == ["x"]
    assert joint.metadata["overlap_axes"] == ["y", "z"]
    assert joint.metadata["joinery_hint"] == "panel_to_block_fastener"


def test_near_regions_join_within_tolerance_and_far_regions_do_not():
    regions = [
        _region("left", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
        _region("near", (12.0, 0.0, 0.0), (22.0, 10.0, 10.0)),
        _region("far", (30.0, 0.0, 0.0), (40.0, 10.0, 10.0)),
    ]
    assignments = [
        _assignment("a_left", "left", "contour_stack", ["part_left"]),
        _assignment("a_near", "near", "voxel_blocks", ["part_near"]),
        _assignment("a_far", "far", "planar_skin", ["part_far"]),
    ]
    parts = [
        _part("part_left", "contour_stack"),
        _part("part_near", "voxel_blocks"),
        _part("part_far", "planar_skin"),
    ]

    joints = synthesize_hybrid_boundary_joints(
        regions,
        assignments,
        parts,
        max_boundary_distance_mm=2.5,
    )

    assert len(joints) == 1
    assert joints[0].metadata["region_ids"] == ["left", "near"]
    assert joints[0].metadata["boundary_class"] == "near_face"
    assert joints[0].metadata["distance_mm"] == pytest.approx(2.0)
    assert joints[0].metadata["joinery_hint"] == "bridge_gap_with_splice_or_fastener"


def test_skips_assignments_without_composed_parts():
    regions = [
        _region("left", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
        _region("right", (10.0, 0.0, 0.0), (20.0, 10.0, 10.0)),
    ]
    assignments = [
        _assignment("a_left", "left", "contour_stack", ["part_left"]),
        _assignment("a_right", "right", "voxel_blocks", ["missing_part"]),
    ]
    parts = [_part("part_left", "contour_stack")]

    joints = synthesize_hybrid_boundary_joints(regions, assignments, parts)

    assert joints == []


def test_add_hybrid_boundary_joints_is_idempotent_for_generated_joints():
    plan = HybridFabricationPlan(
        status="ok",
        regions=[
            _region("left", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
            _region("right", (10.0, 0.0, 0.0), (20.0, 10.0, 10.0)),
        ],
        assignments=[
            _assignment("a_left", "left", "contour_stack", ["part_left"]),
            _assignment("a_right", "right", "voxel_blocks", ["part_right"]),
        ],
        parts=[
            _part("part_left", "contour_stack"),
            _part("part_right", "voxel_blocks"),
        ],
    )

    returned = add_hybrid_boundary_joints(plan)
    add_hybrid_boundary_joints(plan)

    assert returned is plan
    assert len(plan.joints) == 1
    assert plan.joints[0].metadata["generated_by"] == "fabrication.hybrid_joinery"
