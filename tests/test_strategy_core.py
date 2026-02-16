from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from geometry_primitives import PartProfile2D
from step3_first_principles import (
    ManufacturingPart,
    RegionType,
    _RegionCandidate,
    _collapse_planar_face_pairs,
    _select_candidates,
    _synthesize_joint_specs,
    Step3Input,
    build_default_capability_profile,
    decompose_first_principles,
)


def test_strategy_returns_parts_for_box(box_mesh_file: str):
    spec = Step3Input(
        mesh_path=box_mesh_file,
        design_name="box_core",
        auto_scale=False,
        part_budget_max=6,
        material_preferences=["plywood_baltic_birch"],
        scs_capabilities=build_default_capability_profile(
            material_key="plywood_baltic_birch",
            allow_controlled_bending=False,
        ),
    )
    result = decompose_first_principles(spec)

    assert result.status in {"success", "partial"}
    assert len(result.parts) > 0
    assert result.quality_metrics.part_count == len(result.parts)
    assert "plane_constraint_difficulty_mean" in result.debug
    assert "constraint_difficulty_parts" in result.debug


def test_strategy_no_bending_produces_no_bend_ops(cylinder_mesh_file: str):
    spec = Step3Input(
        mesh_path=cylinder_mesh_file,
        design_name="cylinder_no_bend",
        auto_scale=False,
        part_budget_max=8,
        material_preferences=["plywood_baltic_birch"],
        scs_capabilities=build_default_capability_profile(
            material_key="plywood_baltic_birch",
            allow_controlled_bending=False,
        ),
    )
    result = decompose_first_principles(spec)

    assert all(len(part.bend_ops) == 0 for part in result.parts.values())


def test_strategy_status_fail_for_missing_mesh():
    spec = Step3Input(mesh_path="/tmp/does-not-exist-mesh.stl")

    try:
        decompose_first_principles(spec)
    except FileNotFoundError:
        assert True
    else:
        raise AssertionError("Expected FileNotFoundError for missing mesh")


def test_collapse_planar_face_pairs_merges_opposites():
    profile = PartProfile2D(outline=Polygon([(0, 0), (200, 0), (200, 40), (0, 40)]))
    a = _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=profile,
        normal=np.array([1.0, 0.0, 0.0]),
        position_3d=np.array([10.0, 50.0, 30.0]),
        area_mm2=8000.0,
        source_faces=[1, 2, 3],
        bend_ops=[],
        score=8000.0,
        metadata={"kind": "facet"},
    )
    b = _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=profile,
        normal=np.array([-1.0, 0.0, 0.0]),
        position_3d=np.array([16.0, 50.0, 30.0]),
        area_mm2=8000.0,
        source_faces=[4, 5, 6],
        bend_ops=[],
        score=8000.0,
        metadata={"kind": "facet"},
    )
    c = _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=profile,
        normal=np.array([0.0, 1.0, 0.0]),
        position_3d=np.array([140.0, 10.0, 30.0]),
        area_mm2=8000.0,
        source_faces=[7, 8, 9],
        bend_ops=[],
        score=8000.0,
        metadata={"kind": "facet"},
    )

    merged, pair_count = _collapse_planar_face_pairs([a, b, c])
    assert pair_count == 1
    assert len(merged) == 2
    merged_pair = next(m for m in merged if m.metadata.get("kind") == "merged_facet_pair")
    assert merged_pair.metadata.get("member_thickness_mm") == 6.0


def test_select_candidates_can_allocate_stack_layers_under_budget():
    profile = PartProfile2D(
        outline=Polygon([(0, 0), (300, 0), (300, 100), (0, 100)]),
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
    )
    thick_target = _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=profile,
        normal=np.array([0.0, 0.0, 1.0]),
        position_3d=np.array([0.0, 0.0, 50.0]),
        area_mm2=30000.0,
        source_faces=[0, 1, 2],
        bend_ops=[],
        score=30000.0,
        metadata={"kind": "merged_facet_pair", "member_thickness_mm": 12.7},
    )
    spec = Step3Input(
        mesh_path="/tmp/not-used.stl",
        part_budget_max=2,
        enable_planar_stacking=True,
    )

    parts, debug = _select_candidates([thick_target], spec)

    assert len(parts) == 2
    assert debug["stacked_extra_layers"] == 1
    assert {int(p.metadata["stack_layer_index"]) for p in parts} == {1, 2}
    assert len({str(p.metadata["stack_group_id"]) for p in parts}) == 1


def test_synthesize_joint_specs_skips_same_stack_group_connections():
    profile = PartProfile2D(
        outline=Polygon([(-50, -20), (50, -20), (50, 20), (-50, 20)]),
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
    )
    base = ManufacturingPart(
        part_id="part_00",
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
        profile=profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([0.0, 0.0, 0.0]),
        rotation_3d=np.zeros(3),
        source_area_mm2=4000.0,
        source_faces=[0, 1],
        metadata={"stack_group_id": "cand_0001"},
    )
    stacked = ManufacturingPart(
        part_id="part_01",
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
        profile=profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([0.0, 0.0, 6.35]),
        rotation_3d=np.zeros(3),
        source_area_mm2=4000.0,
        source_faces=[2, 3],
        metadata={"stack_group_id": "cand_0001"},
    )
    other = ManufacturingPart(
        part_id="part_02",
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
        profile=profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([0.0, 0.0, 3.175]),
        rotation_3d=np.array([0.0, np.pi / 2.0, 0.0]),
        source_area_mm2=4000.0,
        source_faces=[4, 5],
        metadata={"stack_group_id": "cand_0002"},
    )

    joints = _synthesize_joint_specs(
        [base, stacked, other],
        joint_distance_mm=100.0,
        contact_tolerance_mm=2.0,
    )
    pairs = {(j.part_a, j.part_b) for j in joints}

    assert ("part_00", "part_01") not in pairs
    assert ("part_00", "part_02") in pairs
    assert ("part_01", "part_02") in pairs


def test_select_candidates_drops_thin_side_caps_when_stack_coverage_is_high():
    top_profile = PartProfile2D(
        outline=Polygon([(0, 0), (600, 0), (600, 300), (0, 300)]),
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
    )
    side_profile = PartProfile2D(
        outline=Polygon([(0, 0), (600, 0), (600, 40), (0, 40)]),
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
    )
    aux_profile = PartProfile2D(
        outline=Polygon([(0, 0), (200, 0), (200, 80), (0, 80)]),
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
    )

    top = _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=top_profile,
        normal=np.array([0.0, 0.0, 1.0]),
        position_3d=np.array([0.0, 0.0, 200.0]),
        area_mm2=180000.0,
        source_faces=[0, 1, 2],
        bend_ops=[],
        score=180000.0,
        metadata={"kind": "merged_facet_pair", "member_thickness_mm": 40.0},
    )
    side_cap = _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=side_profile,
        normal=np.array([0.0, 1.0, 0.0]),
        position_3d=np.array([0.0, 0.0, 20.0]),
        area_mm2=24000.0,
        source_faces=[3, 4, 5],
        bend_ops=[],
        score=24000.0,
        metadata={"kind": "facet_side"},
    )
    aux = _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=aux_profile,
        normal=np.array([1.0, 0.0, 0.0]),
        position_3d=np.array([100.0, 0.0, 80.0]),
        area_mm2=16000.0,
        source_faces=[6, 7, 8],
        bend_ops=[],
        score=16000.0,
        metadata={"kind": "facet_aux"},
    )

    spec = Step3Input(
        mesh_path="/tmp/not-used.stl",
        part_budget_max=5,
        enable_planar_stacking=True,
        max_stack_layers_per_region=4,
        stack_roundup_bias=0.35,
        stack_extra_layer_gain=0.65,
        enable_thin_side_suppression=True,
        thin_side_dim_multiplier=1.10,
        thin_side_aspect_limit=0.25,
        thin_side_coverage_penalty_start=0.40,
        thin_side_coverage_drop_threshold=0.62,
    )

    parts, debug = _select_candidates([top, side_cap, aux], spec)

    assert len(parts) == 5
    assert debug["thin_side_dropped_count"] >= 1
    assert not any(p.metadata.get("kind") == "facet_side" for p in parts)


def test_select_candidates_drops_parallel_overlapping_members():
    centered = PartProfile2D(
        outline=Polygon([(-100, -50), (100, -50), (100, 50), (-100, 50)]),
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
    )
    a = _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=centered,
        normal=np.array([0.0, 0.0, 1.0]),
        position_3d=np.array([0.0, 0.0, 40.0]),
        area_mm2=20000.0,
        source_faces=[0, 1],
        bend_ops=[],
        score=20000.0,
        metadata={"kind": "facet_a"},
    )
    b = _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=centered,
        normal=np.array([0.0, 0.0, 1.0]),
        position_3d=np.array([0.0, 0.0, 40.0]),
        area_mm2=19000.0,
        source_faces=[2, 3],
        bend_ops=[],
        score=19000.0,
        metadata={"kind": "facet_b"},
    )
    spec = Step3Input(
        mesh_path="/tmp/not-used.stl",
        part_budget_max=2,
        enable_planar_stacking=False,
        enable_intersection_filter=True,
        allow_joint_intent_intersections=True,
    )

    parts, debug = _select_candidates([a, b], spec)

    assert len(parts) == 1
    assert debug["intersection_dropped_count"] >= 1


def test_select_candidates_keeps_orthogonal_joint_intent_crossing():
    centered = PartProfile2D(
        outline=Polygon([(-90, -45), (90, -45), (90, 45), (-90, 45)]),
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
    )
    a = _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=centered,
        normal=np.array([0.0, 0.0, 1.0]),
        position_3d=np.array([0.0, 0.0, 30.0]),
        area_mm2=16000.0,
        source_faces=[0, 1],
        bend_ops=[],
        score=16000.0,
        metadata={"kind": "facet_a"},
    )
    b = _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=centered,
        normal=np.array([1.0, 0.0, 0.0]),
        position_3d=np.array([0.0, 0.0, 30.0]),
        area_mm2=15000.0,
        source_faces=[2, 3],
        bend_ops=[],
        score=15000.0,
        metadata={"kind": "facet_b"},
    )
    spec = Step3Input(
        mesh_path="/tmp/not-used.stl",
        part_budget_max=2,
        enable_planar_stacking=False,
        enable_intersection_filter=True,
        allow_joint_intent_intersections=True,
    )

    parts, debug = _select_candidates([a, b], spec)

    assert len(parts) == 2
    assert debug["intersection_allowed_joint_intent_count"] >= 1


def test_select_candidates_reports_constraint_difficulty_metrics():
    profile = PartProfile2D(
        outline=Polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)]),
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
    )
    # Candidate A intentionally has lower source area than its outline area.
    # This simulates an unconstrained plane that needs more trimming to match mesh support.
    a = _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=profile,
        normal=np.array([0.0, 0.0, 1.0]),
        position_3d=np.array([0.0, 0.0, 25.0]),
        area_mm2=6000.0,
        source_faces=[0, 1],
        bend_ops=[],
        score=12000.0,
        metadata={"kind": "facet_a"},
    )
    b = _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=profile,
        normal=np.array([1.0, 0.0, 0.0]),
        position_3d=np.array([0.0, 0.0, 25.0]),
        area_mm2=10000.0,
        source_faces=[2, 3],
        bend_ops=[],
        score=11000.0,
        metadata={"kind": "facet_b"},
    )

    spec = Step3Input(
        mesh_path="/tmp/not-used.stl",
        part_budget_max=2,
        enable_planar_stacking=False,
        enable_intersection_filter=True,
        allow_joint_intent_intersections=True,
    )

    parts, debug = _select_candidates([a, b], spec)

    assert len(parts) == 2
    assert 0.0 < debug["mesh_constraint_pressure_mean"] <= 1.0
    assert 0.0 < debug["intersection_constraint_pressure_mean"] <= 1.0
    assert 0.0 < debug["plane_constraint_difficulty_mean"] <= 1.0
    assert debug["plane_constraint_difficulty_max"] >= debug["plane_constraint_difficulty_mean"]
    assert len(debug["constraint_difficulty_parts"]) == 2
    assert all(
        "plane_constraint_difficulty" in p.metadata and "mesh_constraint_pressure" in p.metadata
        for p in parts
    )


def test_synthesize_joint_specs_requires_geometric_contact_band():
    profile = PartProfile2D(
        outline=Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
    )
    a = ManufacturingPart(
        part_id="part_00",
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
        profile=profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([0.0, 0.0, 0.0]),
        rotation_3d=np.zeros(3),
        source_area_mm2=400.0,
        source_faces=[0],
        metadata={},
    )
    b = ManufacturingPart(
        part_id="part_01",
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
        profile=profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([40.0, 0.0, 0.0]),  # 20mm in-plane gap
        rotation_3d=np.zeros(3),
        source_area_mm2=400.0,
        source_faces=[1],
        metadata={},
    )

    joints = _synthesize_joint_specs(
        [a, b],
        joint_distance_mm=240.0,
        contact_tolerance_mm=2.0,
    )
    assert joints == []


def test_synthesize_joint_specs_skips_parallel_contact_pairs():
    profile = PartProfile2D(
        outline=Polygon([(-25, -25), (25, -25), (25, 25), (-25, 25)]),
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
    )
    a = ManufacturingPart(
        part_id="part_00",
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
        profile=profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([0.0, 0.0, 0.0]),
        rotation_3d=np.zeros(3),
        source_area_mm2=2500.0,
        source_faces=[0],
        metadata={},
    )
    b = ManufacturingPart(
        part_id="part_01",
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
        profile=profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([0.0, 0.0, 6.0]),
        rotation_3d=np.zeros(3),
        source_area_mm2=2500.0,
        source_faces=[1],
        metadata={},
    )

    joints = _synthesize_joint_specs(
        [a, b],
        joint_distance_mm=240.0,
        contact_tolerance_mm=2.0,
        parallel_dot_threshold=0.95,
    )
    assert joints == []
