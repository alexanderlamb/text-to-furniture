"""Tests for joint_synthesizer module."""
import numpy as np
import pytest
from shapely.geometry import Polygon

from geometry_primitives import PartProfile2D, FeatureType
from joint_synthesizer import (
    JointSynthesisConfig,
    JointSpec,
    synthesize_joints,
)
from furniture import JointType


@pytest.fixture
def two_perpendicular_parts():
    """Two rectangular parts that meet at a T-junction."""
    # Horizontal part: 300 x 100
    part_a = PartProfile2D(
        outline=Polygon([(0, 0), (300, 0), (300, 100), (0, 100)]),
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
    )
    # Vertical part: 100 x 200
    part_b = PartProfile2D(
        outline=Polygon([(0, 0), (100, 0), (100, 200), (0, 200)]),
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
    )
    return {"part_a": part_a, "part_b": part_b}


class TestTabSlotJoint:
    """Test tab-slot joint synthesis."""

    def test_creates_tabs_and_slots(self, two_perpendicular_parts):
        parts = two_perpendicular_parts
        config = JointSynthesisConfig(
            tab_width_mm=20.0,
            min_tabs_per_edge=2,
            add_dogbones=True,
            dogbone_radius_mm=1.6,
        )

        specs = [JointSpec(
            part_a_key="part_a",
            part_b_key="part_b",
            joint_type=JointType.TAB_SLOT,
            edge_start=(100.0, 50.0),
            edge_end=(200.0, 50.0),
            mating_thickness_mm=6.35,
        )]

        modified_parts, joints = synthesize_joints(parts, specs, config)

        assert len(joints) == 1
        assert joints[0].joint_type == JointType.TAB_SLOT

        # Part A should have tab features
        tab_features = [
            f for f in modified_parts["part_a"].features
            if f.feature_type == FeatureType.TAB
        ]
        assert len(tab_features) >= 2  # min_tabs_per_edge

        # Part B should have slot cutouts
        assert len(modified_parts["part_b"].cutouts) >= 2

    def test_slots_wider_than_tabs(self, two_perpendicular_parts):
        """Slots should be wider than tabs (slip fit clearance)."""
        parts = two_perpendicular_parts
        config = JointSynthesisConfig(
            tab_width_mm=20.0,
            slip_fit_clearance_mm=0.254,
            add_dogbones=False,
        )

        specs = [JointSpec(
            part_a_key="part_a",
            part_b_key="part_b",
            joint_type=JointType.TAB_SLOT,
            edge_start=(100.0, 50.0),
            edge_end=(200.0, 50.0),
            mating_thickness_mm=6.35,
        )]

        modified_parts, _ = synthesize_joints(parts, specs, config)

        # Check that slot features exist and have width > tab width
        slot_features = [
            f for f in modified_parts["part_b"].features
            if f.feature_type == FeatureType.SLOT
        ]
        if slot_features:
            slot_width = slot_features[0].parameters.get("width_mm", 0)
            assert slot_width > config.tab_width_mm


class TestFingerJoint:
    """Test finger joint synthesis."""

    def test_alternating_fingers(self, two_perpendicular_parts):
        parts = two_perpendicular_parts
        config = JointSynthesisConfig(
            finger_joint_pitch_mm=15.0,
            add_dogbones=True,
        )

        specs = [JointSpec(
            part_a_key="part_a",
            part_b_key="part_b",
            joint_type=JointType.FINGER,
            edge_start=(50.0, 50.0),
            edge_end=(250.0, 50.0),
            mating_thickness_mm=6.35,
        )]

        modified_parts, joints = synthesize_joints(parts, specs, config)

        assert len(joints) == 1
        assert joints[0].joint_type == JointType.FINGER

        # Both parts should have features
        assert len(modified_parts["part_a"].features) > 0
        assert len(modified_parts["part_b"].features) > 0


class TestThroughBolt:
    """Test through-bolt hole synthesis."""

    def test_creates_matching_holes(self, two_perpendicular_parts):
        parts = two_perpendicular_parts
        config = JointSynthesisConfig(bolt_hole_diameter_mm=6.5)

        specs = [JointSpec(
            part_a_key="part_a",
            part_b_key="part_b",
            joint_type=JointType.THROUGH_BOLT,
            edge_start=(50.0, 50.0),
            edge_end=(250.0, 50.0),
            mating_thickness_mm=6.35,
        )]

        modified_parts, joints = synthesize_joints(parts, specs, config)

        assert len(joints) == 1

        # Both parts should have bolt holes
        a_holes = [
            f for f in modified_parts["part_a"].features
            if f.feature_type == FeatureType.BOLT_HOLE
        ]
        b_holes = [
            f for f in modified_parts["part_b"].features
            if f.feature_type == FeatureType.BOLT_HOLE
        ]
        assert len(a_holes) >= 2
        assert len(b_holes) >= 2
        assert len(a_holes) == len(b_holes)


class TestHalfLap:
    """Test half-lap joint synthesis."""

    def test_creates_half_depth_notches(self, two_perpendicular_parts):
        parts = two_perpendicular_parts
        config = JointSynthesisConfig(add_dogbones=True)

        specs = [JointSpec(
            part_a_key="part_a",
            part_b_key="part_b",
            joint_type=JointType.HALF_LAP,
            edge_start=(100.0, 50.0),
            edge_end=(200.0, 50.0),
            mating_thickness_mm=6.35,
        )]

        modified_parts, joints = synthesize_joints(parts, specs, config)

        assert len(joints) == 1
        assert joints[0].joint_type == JointType.HALF_LAP

        # Both parts should have slot cutouts
        assert len(modified_parts["part_a"].cutouts) >= 1
        assert len(modified_parts["part_b"].cutouts) >= 1


class TestMissingParts:
    """Test error handling for invalid joint specs."""

    def test_missing_part_skipped(self):
        parts = {
            "part_a": PartProfile2D(
                outline=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
                thickness_mm=6.35,
            ),
        }
        specs = [JointSpec(
            part_a_key="part_a",
            part_b_key="nonexistent",
            joint_type=JointType.TAB_SLOT,
            edge_start=(0, 50),
            edge_end=(100, 50),
            mating_thickness_mm=6.35,
        )]

        modified_parts, joints = synthesize_joints(parts, specs)
        assert len(joints) == 0
