"""Tests verifying critical bug fixes in the manufacturing pipeline.

Tests for shared modules: scoring (Hausdorff, structural plausibility)
and joint_synthesizer (joint positions).
"""
import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geometry_primitives import PartProfile2D, PlanarPatch, _make_2d_basis
from furniture import ComponentType, FurnitureDesign, Component, Joint, JointType, AssemblyGraph
from joint_synthesizer import JointSpec, JointSynthesisConfig, synthesize_joints
from scoring import compute_hausdorff, _compute_structural_plausibility, _euler_to_rotation_matrix


# ── BUG 4: Hausdorff scoring with rotation ──────────────────────────────────

class TestHausdorffWithRotation:
    """Verify Hausdorff scoring applies part rotation."""

    def test_rotation_matrix_identity(self):
        """Zero rotation should produce identity matrix."""
        rot = _euler_to_rotation_matrix(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-10)

    def test_rotation_matrix_90deg_x(self):
        """90-degree X rotation should swap Y and Z."""
        rot = _euler_to_rotation_matrix(np.array([np.pi / 2, 0.0, 0.0]))
        # After 90deg X rotation, [0,1,0] -> [0,0,1]
        result = rot @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-10)

    def test_rotated_part_different_hausdorff(self):
        """A rotated part should produce different Hausdorff than unrotated."""
        mesh = trimesh.creation.box(extents=[100, 100, 100])
        mesh.apply_translation([0, 0, 50])

        # Unrotated design
        design_a = FurnitureDesign(name="unrotated")
        design_a.add_component(Component(
            name="part_0",
            type=ComponentType.SHELF,
            profile=[(0, 0), (100, 0), (100, 100), (0, 100)],
            thickness=100.0,
            position=np.array([-50.0, -50.0, 0.0]),
            rotation=np.array([0.0, 0.0, 0.0]),
        ))

        # 45-degree rotated design
        design_b = FurnitureDesign(name="rotated")
        design_b.add_component(Component(
            name="part_0",
            type=ComponentType.SHELF,
            profile=[(0, 0), (100, 0), (100, 100), (0, 100)],
            thickness=100.0,
            position=np.array([-50.0, -50.0, 0.0]),
            rotation=np.array([0.0, 0.0, np.pi / 4]),  # 45 deg Z
        ))

        h_a, _ = compute_hausdorff(mesh, design_a, n_points=1000)
        h_b, _ = compute_hausdorff(mesh, design_b, n_points=1000)

        # The rotated version should give a different (likely worse) Hausdorff
        assert h_a != h_b


# ── BUG 5: Joint positions differ on each part ──────────────────────────────

class TestJointPositions:
    """Verify joint positions are different for part_a and part_b."""

    def test_positions_differ_with_basis(self):
        """With basis vectors, position_a and position_b should differ."""
        u_a = np.array([1.0, 0.0, 0.0])
        v_a = np.array([0.0, 1.0, 0.0])
        origin_a = np.array([0.0, 0.0, 500.0])

        u_b = np.array([0.0, 1.0, 0.0])
        v_b = np.array([0.0, 0.0, 1.0])
        origin_b = np.array([150.0, 0.0, 0.0])

        part_a = PartProfile2D(
            outline=Polygon([(0, 0), (300, 0), (300, 200), (0, 200)]),
            basis_u=u_a, basis_v=v_a, origin_3d=origin_a,
        )
        part_b = PartProfile2D(
            outline=Polygon([(0, 0), (200, 0), (200, 500), (0, 500)]),
            basis_u=u_b, basis_v=v_b, origin_3d=origin_b,
        )

        parts = {"part_a": part_a, "part_b": part_b}
        specs = [JointSpec(
            part_a_key="part_a",
            part_b_key="part_b",
            joint_type=JointType.TAB_SLOT,
            edge_start=(100.0, 50.0),
            edge_end=(200.0, 50.0),
            mating_thickness_mm=6.35,
        )]

        _, joints = synthesize_joints(parts, specs)
        assert len(joints) == 1
        # position_a and position_b should be different
        assert joints[0].position_a != joints[0].position_b

    def test_positions_fallback_without_basis(self):
        """Without basis vectors, position_b should use part_b centroid."""
        part_a = PartProfile2D(
            outline=Polygon([(0, 0), (300, 0), (300, 200), (0, 200)]),
        )
        part_b = PartProfile2D(
            outline=Polygon([(0, 0), (200, 0), (200, 500), (0, 500)]),
        )

        parts = {"part_a": part_a, "part_b": part_b}
        specs = [JointSpec(
            part_a_key="part_a",
            part_b_key="part_b",
            joint_type=JointType.TAB_SLOT,
            edge_start=(100.0, 50.0),
            edge_end=(200.0, 50.0),
            mating_thickness_mm=6.35,
        )]

        _, joints = synthesize_joints(parts, specs)
        assert len(joints) == 1
        # position_b should be at part_b's centroid (100, 250)
        assert abs(joints[0].position_b[0] - 100.0) < 1.0
        assert abs(joints[0].position_b[1] - 250.0) < 1.0


# ── BUG 6: Structural plausibility scoring ──────────────────────────────────

class TestStructuralPlausibility:
    """Verify PANEL is not counted as horizontal surface."""

    def test_panel_only_not_horizontal(self):
        """Design with only PANEL should NOT satisfy horizontal check."""
        design = FurnitureDesign(name="panels_only")
        design.add_component(Component(
            name="side_a",
            type=ComponentType.PANEL,
            profile=[(0, 0), (100, 0), (100, 500), (0, 500)],
            thickness=6.35,
        ))
        design.add_component(Component(
            name="side_b",
            type=ComponentType.PANEL,
            profile=[(0, 0), (100, 0), (100, 500), (0, 500)],
            thickness=6.35,
        ))

        plausibility = _compute_structural_plausibility(design)
        # Has vertical (PANEL counts as vertical), but no horizontal
        # So 1 of 4 checks pass (vertical), connectivity partial
        # Score should be < 1.0
        assert plausibility < 1.0

    def test_shelf_is_horizontal(self):
        """Design with SHELF should satisfy horizontal check."""
        design = FurnitureDesign(name="with_shelf")
        design.add_component(Component(
            name="top",
            type=ComponentType.SHELF,
            profile=[(0, 0), (800, 0), (800, 500), (0, 500)],
            thickness=19.0,
        ))
        design.add_component(Component(
            name="leg",
            type=ComponentType.LEG,
            profile=[(0, 0), (40, 0), (40, 40), (0, 40)],
            thickness=680.0,
        ))
        design.assembly.add_joint(Joint(
            component_a="top",
            component_b="leg",
            joint_type=JointType.TAB_SLOT,
            position_a=(20.0, 20.0, 0.0),
            position_b=(0.0, 0.0, 680.0),
        ))

        plausibility = _compute_structural_plausibility(design)
        # Has horizontal (SHELF), vertical (LEG), joints, connected
        assert plausibility == 1.0

    def test_panel_counts_as_vertical(self):
        """PANEL should count as vertical support."""
        design = FurnitureDesign(name="panel_vertical")
        design.add_component(Component(
            name="top",
            type=ComponentType.SHELF,
            profile=[(0, 0), (400, 0), (400, 300), (0, 300)],
            thickness=19.0,
        ))
        design.add_component(Component(
            name="side",
            type=ComponentType.PANEL,
            profile=[(0, 0), (300, 0), (300, 500), (0, 500)],
            thickness=6.35,
        ))
        design.assembly.add_joint(Joint(
            component_a="top",
            component_b="side",
            joint_type=JointType.TAB_SLOT,
            position_a=(0.0, 150.0, 0.0),
            position_b=(150.0, 0.0, 0.0),
        ))

        plausibility = _compute_structural_plausibility(design)
        # Has horizontal (SHELF), vertical (PANEL), joints, connected → 1.0
        assert plausibility == 1.0
