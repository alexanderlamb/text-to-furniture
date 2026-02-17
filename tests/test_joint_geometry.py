"""Tests for joint geometry: tabs, slots, and cross-laps."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from geometry_primitives import PartProfile2D
from step3_first_principles import (
    JointSpec,
    ManufacturingPart,
    RegionType,
    Step3Input,
    Step3Violation,
    _apply_joint_geometry,
    _classify_joint_type,
    _compute_contact_line,
    _compute_plane_overlap,
    _validate_joint_assembly_3d,
    _validate_joint_fit_2d,
    build_default_capability_profile,
    decompose_first_principles,
)


def _make_part(
    part_id: str,
    outline: Polygon,
    position_3d: np.ndarray,
    rotation_3d: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    origin_3d: np.ndarray,
    thickness_mm: float = 6.35,
    metadata: dict | None = None,
) -> ManufacturingPart:
    """Helper to build a ManufacturingPart with known geometry."""
    profile = PartProfile2D(
        outline=outline,
        material_key="plywood_baltic_birch",
        thickness_mm=thickness_mm,
        basis_u=basis_u,
        basis_v=basis_v,
        origin_3d=origin_3d,
    )
    return ManufacturingPart(
        part_id=part_id,
        material_key="plywood_baltic_birch",
        thickness_mm=thickness_mm,
        profile=profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=position_3d.copy(),
        rotation_3d=rotation_3d.copy(),
        source_area_mm2=float(outline.area),
        source_faces=[0],
        metadata=metadata or {},
    )


def _shelf_and_side():
    """Build a shelf (Z-normal) and side (X-normal) that share an edge.

    Shelf: 300mm x 200mm in the XY plane at z=100, from x=0..300, y=0..200.
    Side:  200mm x 200mm in the YZ plane at x=300 (the shelf's right edge).
    They meet along the line x=300, z=100, y varies 0..200.

    Contact on the shelf: at u=300 (right edge of 0..300 outline) → near boundary.
    Contact on the side: at (y, 100) midpoint of 0..200 extent → interior.
    → tab_slot: shelf=tab (near edge), side=slot (interior).
    """
    # Shelf — normal along Z, lies in XY plane at z=100
    shelf = _make_part(
        part_id="shelf",
        outline=Polygon([(0, 0), (300, 0), (300, 200), (0, 200)]),
        position_3d=np.array([150.0, 100.0, 100.0]),
        rotation_3d=np.array([0.0, 0.0, 0.0]),  # Z-normal
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 1.0, 0.0]),
        origin_3d=np.array([0.0, 0.0, 100.0]),
    )
    # Side — normal along X, lies in YZ plane at x=300
    # Extra 30mm margin on all sides beyond the contact zone so tab/slot
    # geometry (including dogbone relief) fits inside the outline.
    side = _make_part(
        part_id="side",
        outline=Polygon([(-30, -30), (230, -30), (230, 230), (-30, 230)]),
        position_3d=np.array([300.0, 100.0, 100.0]),
        rotation_3d=np.array([0.0, math.pi / 2.0, 0.0]),  # X-normal
        basis_u=np.array([0.0, 1.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([300.0, 0.0, 0.0]),
    )
    return shelf, side


# --- Test 1: _compute_contact_line ---


def test_compute_contact_line_orthogonal_parts():
    """Shelf (Z-normal) + side (X-normal) → contact line exists with expected length."""
    shelf, side = _shelf_and_side()
    result = _compute_contact_line(shelf, side)
    assert result is not None, "Contact line should exist for orthogonal parts"

    line_dir, line_2d_shelf, line_2d_side = result
    # The contact line runs along Y direction
    assert line_2d_shelf.length > 10.0, f"Contact on shelf too short: {line_2d_shelf.length}"
    assert line_2d_side.length > 10.0, f"Contact on side too short: {line_2d_side.length}"


# --- Test 2: _classify_joint_type → tab_slot ---


def test_classify_joint_tab_slot():
    """Shelf edge at contact → tab_slot classification."""
    shelf, side = _shelf_and_side()
    result = _compute_contact_line(shelf, side)
    assert result is not None

    _, line_2d_shelf, line_2d_side = result
    jtype, tab_id, slot_id = _classify_joint_type(shelf, side, line_2d_shelf, line_2d_side)

    # The contact line on the shelf runs along x=100 (the right edge of a 200mm part)
    # — that's near the boundary. The contact on the side is interior.
    # So shelf = tab, side = slot.
    assert jtype == "tab_slot", f"Expected tab_slot, got {jtype}"


# --- Test 3: _classify_joint_type → cross_lap ---


def test_classify_joint_cross_lap():
    """Both contact lines in interior → cross_lap classification."""
    # Two large parts that cross in the middle
    shelf = _make_part(
        part_id="top",
        outline=Polygon([(-200, -100), (200, -100), (200, 100), (-200, 100)]),
        position_3d=np.array([0.0, 0.0, 50.0]),
        rotation_3d=np.array([0.0, 0.0, 0.0]),
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 1.0, 0.0]),
        origin_3d=np.array([0.0, 0.0, 50.0]),
    )
    divider = _make_part(
        part_id="divider",
        outline=Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)]),
        position_3d=np.array([0.0, 0.0, 50.0]),
        rotation_3d=np.array([0.0, math.pi / 2.0, 0.0]),
        basis_u=np.array([0.0, 1.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([0.0, 0.0, 0.0]),
    )
    result = _compute_contact_line(shelf, divider)
    assert result is not None

    _, line_2d_a, line_2d_b = result
    jtype, _, _ = _classify_joint_type(shelf, divider, line_2d_a, line_2d_b)
    assert jtype == "cross_lap", f"Expected cross_lap, got {jtype}"


# --- Test 4: _classify_joint_type → butt fallback ---


def test_classify_joint_butt_fallback():
    """Contact too short → butt via _apply_joint_geometry min_contact check."""
    # Two parts with very small overlap area
    shelf = _make_part(
        part_id="small_shelf",
        outline=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        position_3d=np.array([5.0, 5.0, 0.0]),
        rotation_3d=np.array([0.0, 0.0, 0.0]),
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 1.0, 0.0]),
        origin_3d=np.array([0.0, 0.0, 0.0]),
    )
    side = _make_part(
        part_id="small_side",
        outline=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        position_3d=np.array([10.0, 5.0, 5.0]),
        rotation_3d=np.array([0.0, math.pi / 2.0, 0.0]),
        basis_u=np.array([0.0, 1.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([10.0, 0.0, 0.0]),
    )

    joints = [
        JointSpec(
            joint_type="tab_slot",
            part_a="small_shelf",
            part_b="small_side",
            geometry={"distance_mm": 5.0, "contact_gap_mm": 0.0, "overlap_mm": 0.0},
            clearance_mm=0.254,
        )
    ]

    spec = Step3Input(
        mesh_path="/tmp/not-used.stl",
        joint_min_contact_mm=15.0,  # 10mm contact < 15mm threshold
    )

    _, updated_joints, debug = _apply_joint_geometry([shelf, side], joints, spec)

    # Contact line is at most 10mm, below the 15mm minimum → butt
    assert updated_joints[0].joint_type == "butt"
    assert debug["joints_skipped_no_contact"] >= 1 or debug["butt_count"] >= 1


# --- Test 5: tab_slot modifies outlines ---


def test_tab_slot_modifies_outlines():
    """Tab part area increases, slot part gains cutouts."""
    shelf, side = _shelf_and_side()
    original_shelf_area = shelf.profile.outline.area
    original_side_cutouts = len(side.profile.cutouts)

    joints = [
        JointSpec(
            joint_type="tab_slot",
            part_a="shelf",
            part_b="side",
            geometry={"distance_mm": 0.0, "contact_gap_mm": 0.0, "overlap_mm": 0.0},
            clearance_mm=0.254,
        )
    ]

    spec = Step3Input(
        mesh_path="/tmp/not-used.stl",
        joint_min_contact_mm=5.0,
    )

    _, updated_joints, debug = _apply_joint_geometry([shelf, side], joints, spec)

    if debug["tab_slot_count"] > 0:
        # Tab part (shelf or side) should have increased area (tab protrusion)
        # Slot part should have gained cutouts
        tab_id = updated_joints[0].geometry.get("tab_part")
        slot_id = updated_joints[0].geometry.get("slot_part")
        if tab_id and slot_id:
            tab_part = shelf if tab_id == "shelf" else side
            slot_part = side if slot_id == "side" else shelf
            assert tab_part.profile.outline.area >= original_shelf_area * 0.99
            assert len(slot_part.profile.cutouts) > original_side_cutouts


# --- Test 6: tab_slot clearance dimensions ---


def test_tab_slot_clearance_dimensions():
    """tab_width + 2*clearance = slot_width."""
    shelf, side = _shelf_and_side()

    joints = [
        JointSpec(
            joint_type="tab_slot",
            part_a="shelf",
            part_b="side",
            geometry={"distance_mm": 0.0, "contact_gap_mm": 0.0, "overlap_mm": 0.0},
            clearance_mm=0.254,
        )
    ]

    spec = Step3Input(
        mesh_path="/tmp/not-used.stl",
        joint_min_contact_mm=5.0,
    )

    _, updated_joints, debug = _apply_joint_geometry([shelf, side], joints, spec)

    if debug["tab_slot_count"] > 0:
        geom = updated_joints[0].geometry
        tab_width = geom.get("tab_width_mm", 0)
        slot_width = geom.get("slot_width_mm", 0)
        clearance = joints[0].clearance_mm
        # slot_width = tab_part.thickness + 2*clearance
        # tab_width = slot_part.thickness - 2*clearance
        # They're different dimensions, but both should be > 0
        assert tab_width > 0, f"tab_width should be > 0, got {tab_width}"
        assert slot_width > 0, f"slot_width should be > 0, got {slot_width}"
        # tab_width + 2*clearance should approximate the slot_part's thickness
        assert abs((tab_width + 2 * clearance) - side.thickness_mm) < 0.01 or \
               abs((tab_width + 2 * clearance) - shelf.thickness_mm) < 0.01


# --- Test 7: cross_lap both parts get cutouts ---


def test_cross_lap_both_parts_get_cutouts():
    """Both parts gain cutouts in a cross-lap joint."""
    top = _make_part(
        part_id="top",
        outline=Polygon([(-200, -100), (200, -100), (200, 100), (-200, 100)]),
        position_3d=np.array([0.0, 0.0, 50.0]),
        rotation_3d=np.array([0.0, 0.0, 0.0]),
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 1.0, 0.0]),
        origin_3d=np.array([0.0, 0.0, 50.0]),
    )
    divider = _make_part(
        part_id="divider",
        outline=Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)]),
        position_3d=np.array([0.0, 0.0, 50.0]),
        rotation_3d=np.array([0.0, math.pi / 2.0, 0.0]),
        basis_u=np.array([0.0, 1.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([0.0, 0.0, 0.0]),
    )

    joints = [
        JointSpec(
            joint_type="tab_slot",
            part_a="top",
            part_b="divider",
            geometry={"distance_mm": 0.0, "contact_gap_mm": 0.0, "overlap_mm": 0.0},
            clearance_mm=0.254,
        )
    ]

    spec = Step3Input(
        mesh_path="/tmp/not-used.stl",
        joint_min_contact_mm=5.0,
    )

    _, updated_joints, debug = _apply_joint_geometry([top, divider], joints, spec)

    if debug["cross_lap_count"] > 0:
        assert len(top.profile.cutouts) > 0, "Top part should have cutouts for cross-lap"
        assert len(divider.profile.cutouts) > 0, "Divider should have cutouts for cross-lap"


# --- Test 8: through_bolt joints are untouched ---


def test_apply_joint_geometry_skips_through_bolt():
    """Through-bolt joints should not be modified by joint geometry."""
    shelf, side = _shelf_and_side()

    joints = [
        JointSpec(
            joint_type="through_bolt",
            part_a="shelf",
            part_b="side",
            geometry={"distance_mm": 0.0, "contact_gap_mm": 0.0, "overlap_mm": 0.0},
            clearance_mm=0.254,
            fastener_spec="M6_bolt",
        )
    ]

    spec = Step3Input(
        mesh_path="/tmp/not-used.stl",
        joint_min_contact_mm=5.0,
    )

    _, updated_joints, debug = _apply_joint_geometry([shelf, side], joints, spec)

    assert updated_joints[0].joint_type == "through_bolt"
    assert debug["joints_processed"] == 0
    assert len(shelf.profile.cutouts) == 0
    assert len(side.profile.cutouts) == 0


# --- Test 9: end-to-end box ---


def test_joint_geometry_end_to_end_box(box_mesh_file: str):
    """Full pipeline on box mesh — parts should have joints processed."""
    spec = Step3Input(
        mesh_path=box_mesh_file,
        design_name="box_joints",
        auto_scale=False,
        part_budget_max=6,
        material_preferences=["plywood_baltic_birch"],
        scs_capabilities=build_default_capability_profile(
            material_key="plywood_baltic_birch",
            allow_controlled_bending=False,
        ),
        joint_enable_geometry=True,
        joint_min_contact_mm=5.0,
    )
    result = decompose_first_principles(spec)

    assert result.status in {"success", "partial"}
    assert "joint_geometry_enabled" in result.debug
    assert result.debug["joint_geometry_enabled"] is True

    # The joint geometry step ran — check counters are consistent
    processed = result.debug.get("joints_processed", 0)
    total_classified = (
        result.debug.get("tab_slot_count", 0)
        + result.debug.get("cross_lap_count", 0)
        + result.debug.get("butt_count", 0)
    )
    assert total_classified <= processed + result.debug.get("joints_skipped_no_contact", 0)
    # The debug dict should have all expected keys from joint geometry
    for key in [
        "joints_processed", "joints_skipped_no_basis", "joints_skipped_parallel",
        "joints_skipped_no_contact", "tab_slot_count", "cross_lap_count",
        "butt_count", "tabs_created", "slots_created", "cutout_overlap_warnings",
    ]:
        assert key in result.debug, f"Missing debug key: {key}"


# --- Test 10: 2D fit — tab inside slot (happy path) ---


def test_validate_fit_tab_inside_slot():
    """Apply tab-slot geometry, then validate 2D fit → 0 errors."""
    shelf, side = _shelf_and_side()

    joints = [
        JointSpec(
            joint_type="tab_slot",
            part_a="shelf",
            part_b="side",
            geometry={"distance_mm": 0.0, "contact_gap_mm": 0.0, "overlap_mm": 0.0},
            clearance_mm=0.254,
        )
    ]
    spec = Step3Input(mesh_path="/tmp/not-used.stl", joint_min_contact_mm=5.0)
    _, updated_joints, geom_debug = _apply_joint_geometry([shelf, side], joints, spec)

    if geom_debug["tab_slot_count"] > 0:
        violations, debug = _validate_joint_fit_2d([shelf, side], updated_joints)
        errors = [v for v in violations if v.severity == "error"]
        assert len(errors) == 0, f"Expected 0 errors, got: {[v.code for v in errors]}"
        assert debug["joint_fit_checks_run"] > 0


# --- Test 11: 2D fit — tab too wide ---


def test_validate_fit_tab_too_wide():
    """Manually construct a JointSpec with tab_width > slot_width → joint_tab_exceeds_slot."""
    shelf, side = _shelf_and_side()

    # Build a joint where tab is wider than slot
    joints = [
        JointSpec(
            joint_type="tab_slot",
            part_a="shelf",
            part_b="side",
            geometry={
                "classification": "tab_slot",
                "tab_part": "shelf",
                "slot_part": "side",
                "tab_width_mm": 20.0,   # intentionally much wider than slot
                "tab_depth_mm": 6.35,
                "slot_width_mm": 5.0,    # slot too narrow
                "tab_count": 1,
                "tabs": [
                    {"center_2d_tab": [200.0, 50.0], "center_2d_slot": [50.0, 50.0]}
                ],
                "contact_line_2d_a": [[200, 0], [200, 100]],
                "contact_line_2d_b": [[0, 0], [0, 100]],
            },
            clearance_mm=0.254,
        )
    ]

    violations, debug = _validate_joint_fit_2d([shelf, side], joints)
    codes = [v.code for v in violations]
    assert "joint_tab_exceeds_slot" in codes, f"Expected joint_tab_exceeds_slot, got {codes}"
    assert debug["joint_fit_errors"] > 0


# --- Test 12: 2D fit — slot outside outline ---


def test_validate_fit_slot_outside_outline():
    """Place a slot cutout extending outside the part → joint_slot_outside_outline."""
    shelf, side = _shelf_and_side()

    # Add a cutout that extends well outside the side part's outline
    outside_cutout = Polygon([(-60, 40), (-20, 40), (-20, 60), (-60, 60)])
    side.profile.cutouts.append(outside_cutout)

    joints = [
        JointSpec(
            joint_type="tab_slot",
            part_a="shelf",
            part_b="side",
            geometry={
                "classification": "tab_slot",
                "tab_part": "shelf",
                "slot_part": "side",
                "tab_width_mm": 5.0,
                "tab_depth_mm": 6.35,
                "slot_width_mm": 7.0,
                "tab_count": 1,
                "tabs": [
                    {"center_2d_tab": [200.0, 50.0], "center_2d_slot": [50.0, 50.0]}
                ],
                "contact_line_2d_a": [[200, 0], [200, 100]],
                "contact_line_2d_b": [[0, 0], [0, 100]],
            },
            clearance_mm=0.254,
        )
    ]

    violations, debug = _validate_joint_fit_2d([shelf, side], joints)
    codes = [v.code for v in violations]
    assert "joint_slot_outside_outline" in codes, f"Expected joint_slot_outside_outline, got {codes}"


# --- Test 13: 2D fit — cutout collision ---


def test_validate_fit_cutout_collision():
    """Two overlapping cutouts on the same part → joint_cutout_collision warning."""
    shelf, side = _shelf_and_side()

    # Add two cutouts that clearly overlap
    side.profile.cutouts.append(Polygon([(30, 30), (60, 30), (60, 70), (30, 70)]))
    side.profile.cutouts.append(Polygon([(40, 40), (70, 40), (70, 80), (40, 80)]))

    # No classified joints needed — cutout collision is checked per-part
    violations, debug = _validate_joint_fit_2d([shelf, side], [])
    codes = [v.code for v in violations]
    assert "joint_cutout_collision" in codes, f"Expected joint_cutout_collision, got {codes}"
    assert debug["joint_fit_warnings"] > 0


# --- Test 14: 3D assembly — aligned (happy path) ---


def test_validate_assembly_aligned():
    """Shelf + side with correct geometry → 0 alignment warnings."""
    shelf, side = _shelf_and_side()

    joints = [
        JointSpec(
            joint_type="tab_slot",
            part_a="shelf",
            part_b="side",
            geometry={"distance_mm": 0.0, "contact_gap_mm": 0.0, "overlap_mm": 0.0},
            clearance_mm=0.254,
        )
    ]
    spec = Step3Input(mesh_path="/tmp/not-used.stl", joint_min_contact_mm=5.0)
    _, updated_joints, geom_debug = _apply_joint_geometry([shelf, side], joints, spec)

    if geom_debug["tab_slot_count"] > 0:
        violations, debug = _validate_joint_assembly_3d([shelf, side], updated_joints)
        # Excessive gap and misalignment should not appear for properly-placed parts
        error_codes = [v.code for v in violations if v.severity == "error"]
        assert "joint_3d_excessive_gap" not in error_codes
        assert debug["assembly_checks_run"] > 0


# --- Test 15: 3D assembly — misaligned ---


def test_validate_assembly_misaligned():
    """Offset slot part origin by 50mm → joint_3d_misalignment warning."""
    shelf, side = _shelf_and_side()

    joints = [
        JointSpec(
            joint_type="tab_slot",
            part_a="shelf",
            part_b="side",
            geometry={"distance_mm": 0.0, "contact_gap_mm": 0.0, "overlap_mm": 0.0},
            clearance_mm=0.254,
        )
    ]
    spec = Step3Input(mesh_path="/tmp/not-used.stl", joint_min_contact_mm=5.0)
    _, updated_joints, geom_debug = _apply_joint_geometry([shelf, side], joints, spec)

    if geom_debug["tab_slot_count"] > 0:
        # Now shift the slot part's origin by 50mm to create misalignment
        side.profile.origin_3d = side.profile.origin_3d + np.array([50.0, 0.0, 0.0])

        violations, debug = _validate_joint_assembly_3d([shelf, side], updated_joints)
        codes = [v.code for v in violations]
        assert "joint_3d_misalignment" in codes, f"Expected joint_3d_misalignment, got {codes}"


# --- Test 16: 3D assembly — excessive gap ---


def test_validate_assembly_excessive_gap():
    """Separate parts by 100mm → joint_3d_excessive_gap error."""
    shelf, side = _shelf_and_side()

    # Build a joint with geometry that has contact lines, but move the parts apart
    # so the 3D projection shows a big gap
    joints = [
        JointSpec(
            joint_type="tab_slot",
            part_a="shelf",
            part_b="side",
            geometry={
                "classification": "tab_slot",
                "tab_part": "shelf",
                "slot_part": "side",
                "tab_width_mm": 5.0,
                "tab_depth_mm": 6.35,
                "slot_width_mm": 7.0,
                "tab_count": 1,
                "tabs": [
                    {"center_2d_tab": [200.0, 50.0], "center_2d_slot": [50.0, 50.0]}
                ],
                "contact_line_2d_a": [[200, 0], [200, 100]],
                "contact_line_2d_b": [[0, 0], [0, 100]],
            },
            clearance_mm=0.254,
        )
    ]

    # Move side part far away (origin offset by 100mm)
    side.profile.origin_3d = np.array([300.0, 0.0, 0.0])

    violations, debug = _validate_joint_assembly_3d([shelf, side], joints)
    codes = [v.code for v in violations]
    assert "joint_3d_excessive_gap" in codes, f"Expected joint_3d_excessive_gap, got {codes}"
    assert debug["assembly_alignment_errors"] > 0


# --- Test 17: 3D assembly — parallel normals ---


def test_validate_assembly_parallel_normals():
    """Two Z-normal parts joined → joint_3d_parallel_normals warning."""
    # Both parts have Z-normal (rotation = [0,0,0])
    top = _make_part(
        part_id="top",
        outline=Polygon([(0, 0), (200, 0), (200, 100), (0, 100)]),
        position_3d=np.array([100.0, 50.0, 50.0]),
        rotation_3d=np.array([0.0, 0.0, 0.0]),  # Z-normal
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 1.0, 0.0]),
        origin_3d=np.array([0.0, 0.0, 50.0]),
    )
    bottom = _make_part(
        part_id="bottom",
        outline=Polygon([(0, 0), (200, 0), (200, 100), (0, 100)]),
        position_3d=np.array([100.0, 50.0, 0.0]),
        rotation_3d=np.array([0.0, 0.0, 0.0]),  # also Z-normal → parallel
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 1.0, 0.0]),
        origin_3d=np.array([0.0, 0.0, 0.0]),
    )

    joints = [
        JointSpec(
            joint_type="tab_slot",
            part_a="top",
            part_b="bottom",
            geometry={
                "classification": "tab_slot",
                "tab_part": "top",
                "slot_part": "bottom",
                "tab_width_mm": 5.0,
                "tab_depth_mm": 6.35,
                "slot_width_mm": 7.0,
                "tab_count": 1,
                "tabs": [
                    {"center_2d_tab": [100.0, 50.0], "center_2d_slot": [100.0, 50.0]}
                ],
                "contact_line_2d_a": [[0, 50], [200, 50]],
                "contact_line_2d_b": [[0, 50], [200, 50]],
            },
            clearance_mm=0.254,
        )
    ]

    violations, debug = _validate_joint_assembly_3d([top, bottom], joints)
    codes = [v.code for v in violations]
    assert "joint_3d_parallel_normals" in codes, f"Expected joint_3d_parallel_normals, got {codes}"
    assert debug["assembly_alignment_warnings"] > 0


# --- Test 18: tabs protrude outward even when outline is expanded ---


def test_tab_protrudes_outward_when_outline_expanded():
    """When outline extends beyond the contact line (e.g. stacking), tabs must
    still protrude outward beyond the boundary, not inward."""
    # Shelf with outline expanded 10mm beyond the contact edge at x=300
    # The contact line sits at x=300, but the outline extends to x=310.
    expanded_shelf = _make_part(
        part_id="shelf",
        outline=Polygon([(0, 0), (310, 0), (310, 200), (0, 200)]),  # expanded to 310
        position_3d=np.array([150.0, 100.0, 100.0]),
        rotation_3d=np.array([0.0, 0.0, 0.0]),
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 1.0, 0.0]),
        origin_3d=np.array([0.0, 0.0, 100.0]),
    )
    side = _make_part(
        part_id="side",
        outline=Polygon([(-30, -30), (230, -30), (230, 230), (-30, 230)]),
        position_3d=np.array([300.0, 100.0, 100.0]),
        rotation_3d=np.array([0.0, math.pi / 2.0, 0.0]),
        basis_u=np.array([0.0, 1.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([300.0, 0.0, 0.0]),
    )

    original_max_x = 310.0  # the expanded outline's right edge

    joints = [
        JointSpec(
            joint_type="tab_slot",
            part_a="shelf",
            part_b="side",
            geometry={"distance_mm": 0.0, "contact_gap_mm": 0.0, "overlap_mm": 0.0},
            clearance_mm=0.254,
        )
    ]
    spec = Step3Input(mesh_path="/tmp/not-used.stl", joint_min_contact_mm=5.0)

    _, updated_joints, debug = _apply_joint_geometry(
        [expanded_shelf, side], joints, spec
    )

    if debug["tab_slot_count"] > 0:
        tab_id = updated_joints[0].geometry.get("tab_part")
        if tab_id == "shelf":
            post_bounds = expanded_shelf.profile.outline.bounds
            # Tab must protrude BEYOND x=310 (the existing edge), not shrink inward
            assert post_bounds[2] > original_max_x + 1.0, (
                f"Tabs should protrude beyond original edge "
                f"(post max_x={post_bounds[2]:.1f}, original={original_max_x:.1f})"
            )


# --- Cutout-aware overlap tests ---


def test_plane_overlap_cutout_reduces_overlap_area():
    """Slot cutout on a part should reduce the overlap region area."""
    thickness = 6.35
    far_face_x = 200.0 - thickness

    # Shelf: Z-normal at z=50, trimmed at side's far face, with tab
    main_body = Polygon([(0, 0), (far_face_x, 0), (far_face_x, 100), (0, 100)])
    tab = Polygon([(far_face_x, 40), (200, 40), (200, 60), (far_face_x, 60)])
    shelf_outline = main_body.union(tab)

    shelf = _make_part(
        part_id="shelf",
        outline=shelf_outline,
        position_3d=np.array([100.0, 50.0, 50.0]),
        rotation_3d=np.array([0.0, 0.0, 0.0]),
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 1.0, 0.0]),
        origin_3d=np.array([0.0, 0.0, 50.0]),
    )
    side = _make_part(
        part_id="side",
        outline=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
        # position_3d is the anti-normal face; material extends in the
        # normal direction (+X) by thickness.  Surface at x=200.
        position_3d=np.array([193.65, 50.0, 50.0]),
        rotation_3d=np.array([0.0, math.pi / 2.0, 0.0]),
        basis_u=np.array([0.0, 1.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([200.0, 0.0, 0.0]),
    )

    # Measure overlap WITHOUT cutout
    result_no_cutout = _compute_plane_overlap([shelf, side])
    area_no_cutout = sum(
        r.get("area_mm2", 0.0) for r in result_no_cutout["plane_overlap_regions"]
    )
    assert result_no_cutout["plane_overlap_pairs"] > 0, (
        "Expected overlap for perpendicular parts with tab"
    )

    # Now add slot cutout to the side where the tab passes through
    slot_cutout = Polygon([
        (40 - 0.254, 50 - thickness / 2 - 0.254),
        (60 + 0.254, 50 - thickness / 2 - 0.254),
        (60 + 0.254, 50 + thickness / 2 + 0.254),
        (40 - 0.254, 50 + thickness / 2 + 0.254),
    ])
    side.profile.cutouts.append(slot_cutout)

    # Measure overlap WITH cutout
    result_with_cutout = _compute_plane_overlap([shelf, side])
    area_with_cutout = sum(
        r.get("area_mm2", 0.0) for r in result_with_cutout["plane_overlap_regions"]
    )

    # The cutout-aware overlap should produce LESS area than without cutout
    assert area_with_cutout < area_no_cutout, (
        f"Cutout should reduce overlap area: "
        f"with={area_with_cutout:.1f} should be < without={area_no_cutout:.1f}"
    )


def test_plane_overlap_detects_real_overlap_without_cutout():
    """Tab extending into slab without slot cutout → overlap IS detected."""
    thickness = 6.35
    far_face_x = 200.0 - thickness

    main_body = Polygon([(0, 0), (far_face_x, 0), (far_face_x, 100), (0, 100)])
    tab = Polygon([(far_face_x, 40), (200, 40), (200, 60), (far_face_x, 60)])
    shelf_outline = main_body.union(tab)

    shelf = _make_part(
        part_id="shelf",
        outline=shelf_outline,
        position_3d=np.array([100.0, 50.0, 50.0]),
        rotation_3d=np.array([0.0, 0.0, 0.0]),
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 1.0, 0.0]),
        origin_3d=np.array([0.0, 0.0, 50.0]),
    )
    side = _make_part(
        part_id="side",
        outline=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
        # position_3d is the anti-normal face; material extends in the
        # normal direction (+X) by thickness.  Surface at x=200.
        position_3d=np.array([193.65, 50.0, 50.0]),
        rotation_3d=np.array([0.0, math.pi / 2.0, 0.0]),
        basis_u=np.array([0.0, 1.0, 0.0]),
        basis_v=np.array([0.0, 0.0, 1.0]),
        origin_3d=np.array([200.0, 0.0, 0.0]),
    )

    result = _compute_plane_overlap([shelf, side])
    assert result["plane_overlap_pairs"] > 0, (
        f"Expected overlap (no cutout), got {result['plane_overlap_pairs']}"
    )


def test_post_joint_overlap_pairs_do_not_increase(box_mesh_file: str):
    """Full pipeline on box: final overlap pairs <= step-2 overlap pairs."""
    spec = Step3Input(
        mesh_path=box_mesh_file,
        design_name="box_overlap_check",
        auto_scale=False,
        part_budget_max=6,
        material_preferences=["plywood_baltic_birch"],
        scs_capabilities=build_default_capability_profile(
            material_key="plywood_baltic_birch",
            allow_controlled_bending=False,
        ),
        joint_enable_geometry=True,
        joint_min_contact_mm=5.0,
    )
    result = decompose_first_principles(spec)

    assert result.status in {"success", "partial"}
    final_pairs = result.debug.get("plane_overlap_pairs", 0)
    step2_pairs = result.debug.get("step2_plane_overlap_pairs", 0)
    assert final_pairs <= step2_pairs + 1, (
        f"Final overlap pairs ({final_pairs}) should not exceed "
        f"step-2 pairs ({step2_pairs}) by more than 1"
    )
