"""Tests for slab_joints.py — 3D joint detection and 2D projection."""
import numpy as np
import pytest
from shapely.geometry import Polygon

from slab_candidates import Slab3D
from slab_joints import (
    SlabIntersection3D, detect_joints, intersections_to_joint_specs,
    _project_to_slab_2d,
)
from furniture import JointType


def _make_slab(normal, origin, width=200, height=100, thickness=6.35):
    """Helper to create a simple slab with auto-computed basis."""
    from geometry_primitives import _make_2d_basis
    n = np.array(normal, dtype=float)
    n = n / np.linalg.norm(n)
    u, v = _make_2d_basis(n)
    outline = Polygon([
        (-width/2, -height/2),
        (width/2, -height/2),
        (width/2, height/2),
        (-width/2, height/2),
    ])
    return Slab3D(
        normal=n,
        origin=np.array(origin, dtype=float),
        thickness_mm=thickness,
        width_mm=width,
        height_mm=height,
        basis_u=u,
        basis_v=v,
        outline_2d=outline,
        material_key="plywood_baltic_birch",
        source="test",
    )


class TestDetectJoints:
    def test_perpendicular_slabs_produce_tab_slot(self):
        """Two perpendicular slabs should produce TAB_SLOT."""
        slab_a = _make_slab([0, 0, 1], [0, 0, 0], width=200, height=200)
        slab_b = _make_slab([1, 0, 0], [0, 0, 0], width=200, height=200)
        joints = detect_joints([slab_a, slab_b], proximity_mm=50.0)
        assert len(joints) > 0
        assert joints[0].joint_type == JointType.TAB_SLOT

    def test_parallel_slabs_produce_through_bolt(self):
        """Two parallel slabs should produce THROUGH_BOLT."""
        slab_a = _make_slab([0, 0, 1], [0, 0, 0], width=200, height=200)
        slab_b = _make_slab([0, 0, 1], [0, 0, 50], width=200, height=200)
        joints = detect_joints([slab_a, slab_b], proximity_mm=100.0)
        assert len(joints) > 0
        assert joints[0].joint_type == JointType.THROUGH_BOLT

    def test_distant_slabs_no_joint(self):
        """Two slabs far apart should not produce a joint."""
        slab_a = _make_slab([0, 0, 1], [0, 0, 0])
        slab_b = _make_slab([0, 0, 1], [5000, 5000, 5000])
        joints = detect_joints([slab_a, slab_b], proximity_mm=25.0)
        assert len(joints) == 0

    def test_edge_coordinates_in_correct_frames(self):
        """Edge coordinates should be in each slab's local 2D frame."""
        slab_a = _make_slab([0, 0, 1], [0, 0, 0], width=200, height=200)
        slab_b = _make_slab([1, 0, 0], [100, 0, 0], width=200, height=200)

        joints = detect_joints([slab_a, slab_b], proximity_mm=200.0)
        assert len(joints) > 0

        ix = joints[0]
        # Edge on A should be in slab_a's 2D frame
        a_start, a_end = ix.edge_on_a
        # Edge on B should be in slab_b's 2D frame
        b_start, b_end = ix.edge_on_b

        # The edge endpoints projected back to 3D should match
        pt_a_3d = slab_a.origin + slab_a.basis_u * a_start[0] + slab_a.basis_v * a_start[1]
        pt_b_3d = slab_b.origin + slab_b.basis_u * b_start[0] + slab_b.basis_v * b_start[1]

        # Both should project from the same 3D intersection line start
        np.testing.assert_allclose(pt_a_3d, ix.edge_start_3d, atol=1.0)
        np.testing.assert_allclose(pt_b_3d, ix.edge_start_3d, atol=1.0)


class TestIntersectionsToJointSpecs:
    def test_conversion(self):
        slab_a = _make_slab([0, 0, 1], [0, 0, 0])
        slab_b = _make_slab([1, 0, 0], [0, 0, 0])
        joints = detect_joints([slab_a, slab_b], proximity_mm=200.0)
        assert len(joints) > 0

        specs = intersections_to_joint_specs(joints, [slab_a, slab_b])
        assert len(specs) == len(joints)
        assert specs[0].part_a_key == "part_0"
        assert specs[0].part_b_key == "part_1"
        assert specs[0].mating_thickness_mm == slab_b.thickness_mm


class TestMinSegmentLength:
    def test_short_intersection_rejected(self):
        """Two slabs barely touching at a corner should produce no joint."""
        # slab_a is a horizontal panel at z=0, slab_b is a small vertical
        # panel offset so the intersection overlap is only ~10mm along Y,
        # which is below the 15mm threshold.
        slab_a = _make_slab([0, 0, 1], [0, 0, 0], width=200, height=200)
        # Offset slab_b along Y so only ~10mm overlaps with slab_a's extent
        slab_b = _make_slab([1, 0, 0], [0, 95, 0], width=10, height=10)
        joints = detect_joints([slab_a, slab_b], proximity_mm=50.0,
                               min_segment_mm=15.0)
        assert len(joints) == 0

    def test_long_intersection_accepted(self):
        """A substantial overlap should still produce a joint."""
        slab_a = _make_slab([0, 0, 1], [0, 0, 0], width=200, height=200)
        slab_b = _make_slab([1, 0, 0], [0, 0, 0], width=200, height=200)
        joints = detect_joints([slab_a, slab_b], proximity_mm=50.0,
                               min_segment_mm=15.0)
        assert len(joints) > 0


class TestProjectToSlab2D:
    def test_origin_projects_to_zero(self):
        slab = _make_slab([0, 0, 1], [10, 20, 30])
        u, v = _project_to_slab_2d(slab, np.array([10, 20, 30]))
        assert abs(u) < 1e-6
        assert abs(v) < 1e-6

    def test_offset_along_u(self):
        slab = _make_slab([0, 0, 1], [0, 0, 0])
        # Move 50 along basis_u
        pt = slab.origin + slab.basis_u * 50.0
        u, v = _project_to_slab_2d(slab, pt)
        assert abs(u - 50.0) < 1e-6
        assert abs(v) < 1e-6
