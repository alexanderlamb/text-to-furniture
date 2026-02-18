from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import Polygon

from geometry_primitives import PartProfile2D
from step3_first_principles import (
    ManufacturingPart,
    RegionType,
    _RegionCandidate,
    _classify_face_exterior,
    _collapse_planar_face_pairs,
    _enforce_zero_overlap,
    _compute_plane_overlap,
    _identify_joint_contact_pairs,
    _select_candidates,
    _slab_intrusion_polygon,
    _stack_layer_offsets_mm,
    _synthesize_joint_specs,
    _trim_by_slab_subtraction,
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
    merged_pair = next(
        m for m in merged if m.metadata.get("kind") == "merged_facet_pair"
    )
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
    assert (
        debug["plane_constraint_difficulty_max"]
        >= debug["plane_constraint_difficulty_mean"]
    )
    assert len(debug["constraint_difficulty_parts"]) == 2
    assert all(
        "plane_constraint_difficulty" in p.metadata
        and "mesh_constraint_pressure" in p.metadata
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


# --- Exterior-flush stack alignment tests ---


def test_classify_face_exterior_on_box():
    """Faces of a trimesh box with outward normals should classify as exterior."""
    mesh = trimesh.creation.box(extents=[100, 80, 60])
    mesh.fix_normals()
    # Test each unique face normal direction of the box
    for face_idx in range(len(mesh.faces)):
        centroid = mesh.triangles_center[face_idx]
        normal = mesh.face_normals[face_idx]
        assert _classify_face_exterior(
            mesh, centroid, normal
        ), f"Face {face_idx} with normal {normal} should be exterior"


def test_collapse_pairs_without_mesh_defaults_centered():
    """Calling _collapse_planar_face_pairs without mesh gives centered alignment."""
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

    merged, pair_count = _collapse_planar_face_pairs([a, b])
    assert pair_count == 1
    merged_cand = merged[0]
    assert merged_cand.metadata.get("stack_alignment") == "centered"
    assert "exterior_face_origin_3d" not in merged_cand.metadata
    assert "exterior_face_normal" not in merged_cand.metadata


def test_centered_alignment_uses_midpoint():
    """For a merged pair in centered mode, stack centers around position_3d (midpoint)."""
    profile = PartProfile2D(
        outline=Polygon([(0, 0), (300, 0), (300, 100), (0, 100)]),
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
    )
    # Two faces at x=0 and x=20, midpoint at x=10
    cand = _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=profile,
        normal=np.array([1.0, 0.0, 0.0]),
        position_3d=np.array([10.0, 50.0, 30.0]),  # midpoint
        area_mm2=30000.0,
        source_faces=[0, 1, 2],
        bend_ops=[],
        score=30000.0,
        metadata={
            "kind": "merged_facet_pair",
            "member_thickness_mm": 20.0,
            "stack_alignment": "centered",
        },
        origin_3d=np.array([0.0, 50.0, 30.0]),  # different from position_3d
    )
    spec = Step3Input(
        mesh_path="/tmp/not-used.stl",
        part_budget_max=3,
        enable_planar_stacking=True,
    )

    parts, debug = _select_candidates([cand], spec)

    assert len(parts) >= 2
    # All parts should be centered around x=10.0 (position_3d), not x=0.0 (origin_3d)
    positions_x = [p.position_3d[0] for p in parts]
    mean_x = sum(positions_x) / len(positions_x)
    assert (
        abs(mean_x - 10.0) < 1.0
    ), f"Mean x position {mean_x} should be near midpoint 10.0, not origin 0.0"


def test_exterior_flush_outermost_sheet_at_surface():
    """For exterior-flush, sheet 0's outer surface is flush with the exterior face."""
    thickness = 6.35
    profile = PartProfile2D(
        outline=Polygon([(0, 0), (300, 0), (300, 100), (0, 100)]),
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
    )
    ext_origin = np.array([50.0, 0.0, 0.0])
    ext_normal = np.array([1.0, 0.0, 0.0])
    cand = _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=profile,
        normal=np.array([1.0, 0.0, 0.0]),
        position_3d=np.array([40.0, 0.0, 0.0]),  # midpoint
        area_mm2=30000.0,
        source_faces=[0, 1, 2],
        bend_ops=[],
        score=30000.0,
        metadata={
            "kind": "merged_facet_pair",
            "member_thickness_mm": 20.0,
            "stack_alignment": "exterior_flush",
            "exterior_face_origin_3d": [float(v) for v in ext_origin],
            "exterior_face_normal": [float(v) for v in ext_normal],
        },
        origin_3d=np.array([50.0, 0.0, 0.0]),
    )
    spec = Step3Input(
        mesh_path="/tmp/not-used.stl",
        part_budget_max=3,
        enable_planar_stacking=True,
    )

    parts, debug = _select_candidates([cand], spec)

    assert len(parts) >= 2
    # Sheet 0 center should be at ext_origin - 0.5 * thickness * ext_normal
    expected_center_x = 50.0 - 0.5 * thickness
    sheet0 = [p for p in parts if p.metadata.get("stack_layer_index") == 1][0]
    assert (
        abs(sheet0.position_3d[0] - expected_center_x) < 0.01
    ), f"Sheet 0 center x={sheet0.position_3d[0]}, expected {expected_center_x}"
    # Sheet 0's outer surface = center + 0.5 * thickness = ext_origin
    outer_surface_x = sheet0.position_3d[0] + 0.5 * thickness
    assert (
        abs(outer_surface_x - 50.0) < 0.01
    ), f"Sheet 0 outer surface x={outer_surface_x}, expected 50.0 (flush with exterior)"


def test_stack_layer_offsets_mm_unchanged():
    """Regression guard: _stack_layer_offsets_mm still produces correct symmetric offsets."""
    # Single layer
    assert _stack_layer_offsets_mm(1, 6.35, None) == [0.0]

    # Two layers, no member thickness constraint
    offsets_2 = _stack_layer_offsets_mm(2, 6.35, None)
    assert len(offsets_2) == 2
    assert abs(offsets_2[0] + offsets_2[1]) < 1e-9  # symmetric around 0
    assert abs(offsets_2[1] - offsets_2[0] - 6.35) < 1e-9  # spaced by thickness

    # Three layers with member thickness clamping
    offsets_3 = _stack_layer_offsets_mm(3, 6.35, 12.7)
    assert len(offsets_3) == 3
    assert abs(offsets_3[0] + offsets_3[2]) < 1e-9  # symmetric
    # All sheet centers should be within member thickness / 2
    for off in offsets_3:
        assert abs(off) <= 6.35 + 1e-9


# --- Trim offset and overlap metric tests ---


def _make_perpendicular_parts():
    """Build a shelf (XZ plane) and a side panel (YZ plane) for trim tests.

    Side panel: normal along +X, surface at x=270, thickness 6.35mm
    Shelf: normal along +Y, spanning x from 0 to 300, y from 0 to 270+6.35
    """
    thickness = 6.35

    # Side panel: lies in YZ plane, surface at x=270
    side_profile = PartProfile2D(
        outline=Polygon([(0, 0), (300, 0), (300, 400), (0, 400)]),
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        basis_u=np.array([0.0, 1.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([270.0, 0.0, 0.0]),
    )
    side = ManufacturingPart(
        part_id="side_panel",
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        profile=side_profile,
        region_type=RegionType.PLANAR_CUT,
        # position_3d is the anti-normal face; material extends in the normal
        # direction (+X) by thickness.  Surface at x=270, far face at x=263.65.
        position_3d=np.array([263.65, 150.0, 200.0]),
        rotation_3d=np.array([0.0, np.pi / 2.0, 0.0]),  # normal along +X
        source_area_mm2=120000.0,
        source_faces=[0, 1],
        metadata={},
    )

    # Shelf: lies in XZ plane, extends past the side panel
    shelf_profile = PartProfile2D(
        outline=Polygon([(0, 0), (300, 0), (300, 400), (0, 400)]),
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([0.0, 150.0, 0.0]),
    )
    shelf = ManufacturingPart(
        part_id="shelf",
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        profile=shelf_profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([150.0, 150.0, 200.0]),
        rotation_3d=np.array([np.pi / 2.0, 0.0, 0.0]),  # normal along +Y
        source_area_mm2=120000.0,
        source_faces=[2, 3],
        metadata={},
    )

    return shelf, side


def test_trim_offset_depends_on_side():
    """Shelf trimmed via slab subtraction should stop at slab boundary."""
    shelf, side = _make_perpendicular_parts()
    thickness = 6.35

    # The shelf extends from x=0 to x=300, side panel slab is at x=263.65..270.
    # _trim_by_slab_subtraction should clip the shelf to stop at the slab edge.
    spec = Step3Input(mesh_path="/tmp/not-used.stl", part_budget_max=2)
    trimmed, _ = _trim_by_slab_subtraction([shelf, side], spec)

    shelf_trimmed = next(p for p in trimmed if p.part_id == "shelf")
    trimmed_bounds = shelf_trimmed.profile.outline.bounds
    max_x = trimmed_bounds[2]
    # Shelf should stop near the slab boundary (263.65 + tolerance)
    assert max_x < 270.0, f"Shelf max_x={max_x} should be less than 270 (slab surface)"
    assert max_x > 260.0, f"Shelf max_x={max_x} unexpectedly small — too much trimmed"


def _make_normal_side_shelf(side):
    """Build a shelf whose origin is on the NORMAL side of the side panel.

    The shelf extends from x=265 to x=400, with origin at x=280.
    The side panel surface is at x=270.  Since 280 >= 270, the shelf
    centroid is on the normal side, which is the scenario affected by the
    trim-offset bug.
    """
    thickness = 6.35
    shelf_profile = PartProfile2D(
        outline=Polygon([(-15, 0), (120, 0), (120, 400), (-15, 400)]),
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([280.0, 150.0, 0.0]),
    )
    shelf = ManufacturingPart(
        part_id="shelf_normal_side",
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        profile=shelf_profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([332.5, 150.0, 200.0]),
        rotation_3d=np.array([np.pi / 2.0, 0.0, 0.0]),
        source_area_mm2=54000.0,
        source_faces=[10, 11],
        metadata={},
    )
    return shelf


def test_plane_overlap_metric_zero_after_clean_trim():
    """After slab-subtraction trimming, parts should not overlap."""
    _, side = _make_perpendicular_parts()
    shelf = _make_normal_side_shelf(side)

    spec = Step3Input(mesh_path="/tmp/not-used.stl", part_budget_max=2)
    trimmed, _ = _trim_by_slab_subtraction([shelf, side], spec)

    result = _compute_plane_overlap(trimmed)
    assert (
        result["plane_overlap_pairs"] == 0
    ), f"Expected no overlap pairs after trim, got {result}"
    assert result["plane_overlap_max_mm"] < 0.1


def test_trim_by_slab_subtraction_reports_parallel_worker_count():
    shelf, side = _make_perpendicular_parts()
    spec = Step3Input(
        mesh_path="/tmp/not-used.stl",
        part_budget_max=2,
        trim_parallel_workers=2,
    )
    _trimmed, debug = _trim_by_slab_subtraction([shelf, side], spec)
    assert int(debug.get("trim_parallel_workers", 0)) >= 1


def test_strict_no_overlap_prunes_when_trim_budget_is_too_low():
    shelf, side = _make_perpendicular_parts()
    spec = Step3Input(
        mesh_path="/tmp/not-used.stl",
        part_budget_max=2,
        enforce_zero_overlap=True,
        trim_loss_budget_fraction=0.05,
        strict_overlap_max_passes=1,
        strict_overlap_allow_pruning=True,
        strict_overlap_max_part_drops=1,
    )
    cleaned_parts, cleaned_joints, debug = _enforce_zero_overlap(
        [shelf, side], [], spec
    )
    overlap = _compute_plane_overlap(cleaned_parts, joints=cleaned_joints)

    assert debug.get("strict_no_overlap_resolved") is True
    assert overlap["plane_overlap_pairs"] == 0
    assert int(len(cleaned_parts)) == 1
    assert int(len(debug.get("strict_overlap_dropped_parts", []))) == 1


def test_plane_overlap_metric_detects_penetration():
    """A shelf extending past the surface into the slab should be detected."""
    _, side = _make_perpendicular_parts()

    # Build a shelf on the normal side that extends 3mm past the surface
    # into the slab.  Origin at x=280, outline min u = -13 → x = 267.
    # Side panel surface at x=270, so vertex at x=267 is 3mm inside the slab.
    thickness = 6.35
    shelf_profile = PartProfile2D(
        outline=Polygon([(-13, 0), (120, 0), (120, 400), (-13, 400)]),
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([280.0, 150.0, 0.0]),
    )
    shelf = ManufacturingPart(
        part_id="shelf_penetrating",
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        profile=shelf_profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([333.5, 150.0, 200.0]),
        rotation_3d=np.array([np.pi / 2.0, 0.0, 0.0]),
        source_area_mm2=53200.0,
        source_faces=[12, 13],
        metadata={},
    )

    result = _compute_plane_overlap([shelf, side])
    assert (
        result["plane_overlap_pairs"] > 0
    ), f"Expected overlap detection, got {result}"
    # Vertex at x=267 is 3mm past the surface (d=-3), so penetration ≈ 3mm
    assert result["plane_overlap_max_mm"] >= 2.5


def test_plane_overlap_metric_detects_edge_only_crossing():
    """Detect overlap even when no outline vertex lies inside slab interior."""
    _, side = _make_perpendicular_parts()
    thickness = 6.35

    # Shelf spans x=260..300 with origin x=280 (u range -20..20).
    # Relative to side surface x=270, vertices are at d=-10 and d=+30:
    # no vertex lies in slab interior (-6.35, 0), but the edge crosses it.
    shelf_profile = PartProfile2D(
        outline=Polygon([(-20, 0), (20, 0), (20, 400), (-20, 400)]),
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([280.0, 150.0, 0.0]),
    )
    shelf = ManufacturingPart(
        part_id="shelf_edge_crossing",
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        profile=shelf_profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([280.0, 150.0, 200.0]),
        rotation_3d=np.array([np.pi / 2.0, 0.0, 0.0]),
        source_area_mm2=16000.0,
        source_faces=[14, 15],
        metadata={},
    )

    result = _compute_plane_overlap([shelf, side])
    assert (
        result["plane_overlap_pairs"] > 0
    ), f"Expected overlap from edge crossing, got {result}"
    assert result["plane_overlap_max_mm"] >= 3.0


# --- ONE_ZERO trim direction fix tests ---


def test_phantom_pairs_not_trimmed():
    """Phantom pair: parts on non-overlapping planes → no mutual slab penetration → skip."""
    thickness = 6.35

    # Side panel: X-normal at x=200, extends y=0..80, z=0..300
    # The shelf is at y=100 which is outside the side panel's y range (0..80).
    # Mutual slab penetration should be ~0 because the side panel's slab
    # at x=193.65..200 doesn't reach the shelf's y=100 position.
    side_profile = PartProfile2D(
        outline=Polygon([(0, 0), (80, 0), (80, 300), (0, 300)]),
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        basis_u=np.array([0.0, 1.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([200.0, 0.0, 0.0]),
    )
    side = ManufacturingPart(
        part_id="side",
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        profile=side_profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([200.0, 40.0, 150.0]),
        rotation_3d=np.array([0.0, np.pi / 2.0, 0.0]),
        source_area_mm2=24000.0,
        source_faces=[0, 1],
        metadata={},
    )

    # Shelf: Y-normal at y=100, extends x=0..210 (10mm past side surface at x=200).
    shelf_profile = PartProfile2D(
        outline=Polygon([(0, 0), (210, 0), (210, 300), (0, 300)]),
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([0.0, 100.0, 0.0]),
    )
    shelf = ManufacturingPart(
        part_id="shelf",
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        profile=shelf_profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([105.0, 100.0, 150.0]),
        rotation_3d=np.array([-np.pi / 2.0, 0.0, 0.0]),  # Y-normal
        source_area_mm2=63000.0,
        source_faces=[2, 3],
        metadata={},
    )

    spec = Step3Input(mesh_path="/tmp/not-used.stl", part_budget_max=2)
    parts = [shelf, side]
    trimmed, trim_debug = _trim_by_slab_subtraction(parts, spec)

    # Mutual penetration gate should skip this phantom pair entirely.
    # Neither part should be trimmed.
    shelf_trimmed = next(p for p in trimmed if p.part_id == "shelf")
    side_trimmed = next(p for p in trimmed if p.part_id == "side")

    shelf_max_x = shelf_trimmed.profile.outline.bounds[2]
    assert (
        shelf_max_x >= 209.0
    ), f"Shelf should NOT be trimmed (phantom pair), got max_x={shelf_max_x}"

    side_area = side_trimmed.profile.outline.area
    assert side_area >= 23000.0, f"Side should NOT be trimmed, area={side_area}"


def test_slab_intrusion_empty_for_distant_parts():
    """_slab_intrusion_polygon returns empty for parts that are physically separated."""
    thickness = 6.35

    # Side panel at x=300
    side_profile = PartProfile2D(
        outline=Polygon([(0, 0), (100, 0), (100, 200), (0, 200)]),
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        basis_u=np.array([0.0, 1.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([300.0, 0.0, 0.0]),
    )
    side = ManufacturingPart(
        part_id="side",
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        profile=side_profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([300.0, 50.0, 100.0]),
        rotation_3d=np.array([0.0, np.pi / 2.0, 0.0]),
        source_area_mm2=20000.0,
        source_faces=[0],
        metadata={},
    )

    # Shelf at y=200, only extends to x=150 — far from side at x=300
    shelf_profile = PartProfile2D(
        outline=Polygon([(0, 0), (150, 0), (150, 200), (0, 200)]),
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([0.0, 200.0, 0.0]),
    )
    shelf = ManufacturingPart(
        part_id="shelf",
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        profile=shelf_profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([75.0, 200.0, 100.0]),
        rotation_3d=np.array([np.pi / 2.0, 0.0, 0.0]),
        source_area_mm2=30000.0,
        source_faces=[1],
        metadata={},
    )

    # Shelf doesn't reach the side panel's slab → no intrusion
    result = _slab_intrusion_polygon(shelf, side)
    assert result == [], f"Expected empty intrusion, got {len(result)} polygons"


def test_slab_intrusion_t_junction_strip():
    """At a T-junction, intrusion should be a thin strip ~thickness wide."""
    thickness = 6.35

    # Side panel: normal +X, surface at x=270, far face at x=263.65
    side_profile = PartProfile2D(
        outline=Polygon([(0, 0), (300, 0), (300, 400), (0, 400)]),
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        basis_u=np.array([0.0, 1.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([270.0, 0.0, 0.0]),
    )
    side = ManufacturingPart(
        part_id="side",
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        profile=side_profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([263.65, 150.0, 200.0]),
        rotation_3d=np.array([0.0, np.pi / 2.0, 0.0]),
        source_area_mm2=120000.0,
        source_faces=[0],
        metadata={},
    )

    # Shelf: normal +Y, extends x=0..300 (past the side panel slab)
    shelf_profile = PartProfile2D(
        outline=Polygon([(0, 0), (300, 0), (300, 400), (0, 400)]),
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([0.0, 150.0, 0.0]),
    )
    shelf = ManufacturingPart(
        part_id="shelf",
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        profile=shelf_profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([150.0, 150.0, 200.0]),
        rotation_3d=np.array([np.pi / 2.0, 0.0, 0.0]),
        source_area_mm2=120000.0,
        source_faces=[1],
        metadata={},
    )

    # Shelf intrusion into side panel's slab should be a thin strip
    intrusion = _slab_intrusion_polygon(shelf, side)
    assert len(intrusion) > 0, "Expected intrusion at T-junction"
    total_area = sum(p.area for p in intrusion)
    # Strip should be roughly thickness * shelf_height (minus tolerance)
    # ~5.35 * 400 = 2140 (with tolerance reducing effective thickness)
    assert total_area > 1000, f"Intrusion area {total_area} too small for T-junction"
    assert (
        total_area < 5000
    ), f"Intrusion area {total_area} too large — not a thin strip"


def test_identify_joint_contact_pairs_finds_perpendicular_contacts():
    """_identify_joint_contact_pairs returns only perpendicular contact pairs."""
    thickness = 6.35

    # Part A: Z-normal at z=50
    profile_a = PartProfile2D(
        outline=Polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)]),
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
    )
    a = ManufacturingPart(
        part_id="part_a",
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        profile=profile_a,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([0.0, 0.0, 50.0]),
        rotation_3d=np.zeros(3),  # Z-normal
        source_area_mm2=10000.0,
        source_faces=[0],
        metadata={},
    )

    # Part B: X-normal at x=50 (perpendicular to A, in contact)
    profile_b = PartProfile2D(
        outline=Polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)]),
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
    )
    b = ManufacturingPart(
        part_id="part_b",
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        profile=profile_b,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([50.0, 0.0, 50.0]),
        rotation_3d=np.array([0.0, np.pi / 2.0, 0.0]),  # X-normal
        source_area_mm2=10000.0,
        source_faces=[1],
        metadata={},
    )

    # Part C: Z-normal at z=200 (parallel to A, far away)
    profile_c = PartProfile2D(
        outline=Polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)]),
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
    )
    c = ManufacturingPart(
        part_id="part_c",
        material_key="plywood_baltic_birch",
        thickness_mm=thickness,
        profile=profile_c,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array([0.0, 0.0, 200.0]),
        rotation_3d=np.zeros(3),  # Z-normal (parallel to A)
        source_area_mm2=10000.0,
        source_faces=[2],
        metadata={},
    )

    pairs = _identify_joint_contact_pairs(
        [a, b, c],
        contact_tolerance_mm=2.0,
        parallel_dot_threshold=0.95,
    )

    # Only the perpendicular contact pair (0, 1) should be found
    assert (0, 1) in pairs, f"Expected (0,1) in pairs, got {pairs}"
    # Parallel pair (0, 2) should NOT be found
    assert (0, 2) not in pairs, f"Parallel pair (0,2) should not be in {pairs}"
    # (1, 2) may or may not be a contact depending on geometry
    # but the key assertion is that perpendicular contacts are found
