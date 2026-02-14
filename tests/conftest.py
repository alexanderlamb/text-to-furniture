"""
Shared test fixtures for manufacturing-aware pipeline tests.
"""
import sys
import warnings
from pathlib import Path

# Suppress trimesh internal RuntimeWarning for degenerate cross-sections
# (divide-by-zero in center_mass when slicing yields zero-volume geometry).
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning,
    module=r"trimesh\.triangles",
)

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dfm_rules import DFMConfig
from geometry_primitives import PartProfile2D, PlanarPatch
from furniture import (
    Component, ComponentType, Joint, JointType,
    AssemblyGraph, FurnitureDesign,
)


@pytest.fixture
def box_mesh():
    """A simple 100x100x100mm box mesh."""
    mesh = trimesh.creation.box(extents=[100, 100, 100])
    # Centre at origin with bottom at z=0
    mesh.apply_translation([0, 0, 50])
    return mesh


@pytest.fixture
def table_mesh():
    """A simple table-like mesh: top + 4 legs.

    Top: 800x500x20 at z=680
    Legs: 40x40x680 at each corner
    """
    top = trimesh.creation.box(extents=[800, 500, 20])
    top.apply_translation([0, 0, 690])

    legs = []
    for x, y in [(-370, -220), (370, -220), (-370, 220), (370, 220)]:
        leg = trimesh.creation.box(extents=[40, 40, 680])
        leg.apply_translation([x, y, 340])
        legs.append(leg)

    mesh = trimesh.util.concatenate([top] + legs)
    return mesh


@pytest.fixture
def cylinder_mesh():
    """A cylinder mesh (height=200, radius=50)."""
    mesh = trimesh.creation.cylinder(radius=50, height=200)
    mesh.apply_translation([0, 0, 100])
    return mesh


@pytest.fixture
def dfm_config():
    """Standard DFM config for tests."""
    return DFMConfig(
        min_slot_width_inch=0.125,
        min_internal_radius_inch=0.063,
        slip_fit_clearance_inch=0.010,
        min_bridge_width_inch=0.125,
        max_aspect_ratio=20.0,
        max_sheet_width_mm=609.6,
        max_sheet_height_mm=762.0,
    )


@pytest.fixture
def simple_rectangle_profile():
    """A 200x100mm rectangle PartProfile2D."""
    outline = Polygon([(0, 0), (200, 0), (200, 100), (0, 100)])
    return PartProfile2D(
        outline=outline,
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
    )


@pytest.fixture
def simple_table_design():
    """A simple FurnitureDesign of a table with top + 4 legs."""
    design = FurnitureDesign(name="test_table")

    # Top
    design.add_component(Component(
        name="top",
        type=ComponentType.SHELF,
        profile=[(0, 0), (800, 0), (800, 500), (0, 500)],
        thickness=19.0,
        position=np.array([0.0, 0.0, 700.0]),
        material="plywood_baltic_birch",
    ))

    # 4 legs
    for i, (x, y) in enumerate([(20, 20), (720, 20), (20, 420), (720, 420)]):
        design.add_component(Component(
            name=f"leg_{i}",
            type=ComponentType.LEG,
            profile=[(0, 0), (60, 0), (60, 60), (0, 60)],
            thickness=700.0,
            position=np.array([float(x), float(y), 0.0]),
            material="plywood_baltic_birch",
        ))
        design.assembly.add_joint(Joint(
            component_a="top",
            component_b=f"leg_{i}",
            joint_type=JointType.TAB_SLOT,
            position_a=(float(x), float(y), 0.0),
            position_b=(0.0, 0.0, 700.0),
        ))

    return design
