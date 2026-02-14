"""Tests for dfm_rules module."""
import math

import numpy as np
import pytest
from shapely.geometry import Polygon, Point

from dfm_rules import (
    DFMConfig,
    DFMViolation,
    check_part_dfm,
    add_dogbone_relief,
)
from geometry_primitives import PartProfile2D


class TestCheckPartDFM:
    """Test DFM validation checks."""

    def test_valid_rectangle_passes(self, simple_rectangle_profile, dfm_config):
        violations = check_part_dfm(simple_rectangle_profile, dfm_config)
        errors = [v for v in violations if v.severity == "error"]
        assert errors == []

    def test_oversized_part_fails(self, dfm_config):
        """Part wider than max sheet size should fail."""
        outline = Polygon([(0, 0), (1000, 0), (1000, 100), (0, 100)])
        profile = PartProfile2D(outline=outline, thickness_mm=6.35)
        violations = check_part_dfm(profile, dfm_config)
        sheet_errors = [v for v in violations if v.rule_name == "sheet_size_width"]
        assert len(sheet_errors) > 0

    def test_narrow_slot_fails(self, dfm_config):
        """Slot narrower than min_slot_width should fail."""
        outline = Polygon([(0, 0), (200, 0), (200, 100), (0, 100)])
        # Very narrow slot (1mm wide)
        cutout = Polygon([(50, 40), (51, 40), (51, 60), (50, 60)])
        profile = PartProfile2D(
            outline=outline,
            cutouts=[cutout],
            thickness_mm=6.35,
        )
        violations = check_part_dfm(profile, dfm_config)
        slot_errors = [v for v in violations if v.rule_name == "slot_width"]
        assert len(slot_errors) > 0

    def test_adequate_slot_passes(self, dfm_config):
        """Slot wider than min_slot_width should pass."""
        outline = Polygon([(0, 0), (200, 0), (200, 100), (0, 100)])
        # 10mm wide slot (> 1/8" = 3.175mm)
        cutout = Polygon([(50, 30), (60, 30), (60, 70), (50, 70)])
        profile = PartProfile2D(
            outline=outline,
            cutouts=[cutout],
            thickness_mm=6.35,
        )
        violations = check_part_dfm(profile, dfm_config)
        slot_errors = [v for v in violations if v.rule_name == "slot_width"]
        assert slot_errors == []

    def test_high_aspect_ratio_warning(self, dfm_config):
        """Very long, thin part should trigger aspect ratio warning."""
        outline = Polygon([(0, 0), (500, 0), (500, 10), (0, 10)])
        profile = PartProfile2D(outline=outline, thickness_mm=6.35)
        violations = check_part_dfm(profile, dfm_config)
        aspect_warnings = [v for v in violations if v.rule_name == "aspect_ratio"]
        assert len(aspect_warnings) > 0

    def test_bridge_width_warning(self, dfm_config):
        """Cutout too close to edge should trigger bridge width violation."""
        outline = Polygon([(0, 0), (200, 0), (200, 100), (0, 100)])
        # Cutout 1mm from edge (< min bridge width)
        cutout = Polygon([(1, 1), (50, 1), (50, 50), (1, 50)])
        profile = PartProfile2D(
            outline=outline,
            cutouts=[cutout],
            thickness_mm=6.35,
        )
        violations = check_part_dfm(profile, dfm_config)
        bridge_violations = [v for v in violations if v.rule_name == "bridge_width"]
        assert len(bridge_violations) > 0

    def test_from_material(self):
        """DFMConfig.from_material sets sheet size from material catalog."""
        config = DFMConfig.from_material("plywood_baltic_birch")
        assert config.max_sheet_width_mm == pytest.approx(24 * 25.4, abs=0.1)
        assert config.max_sheet_height_mm == pytest.approx(30 * 25.4, abs=0.1)


class TestDogboneRelief:
    """Test dogbone corner relief."""

    def test_rectangle_gets_four_dogbones(self):
        """A rectangle with 4 right-angle corners should get 4 dogbones."""
        rect = Polygon([(0, 0), (50, 0), (50, 30), (0, 30)])
        result = add_dogbone_relief(rect, radius_mm=1.6)

        # Result should be larger than original (circles added at corners)
        assert result.area > rect.area

    def test_dogbone_radius_affects_size(self):
        """Larger radius should produce larger result."""
        rect = Polygon([(0, 0), (50, 0), (50, 30), (0, 30)])
        small = add_dogbone_relief(rect, radius_mm=1.0)
        large = add_dogbone_relief(rect, radius_mm=3.0)
        assert large.area > small.area

    def test_obtuse_angles_no_dogbone(self):
        """Obtuse angles should not get dogbone relief."""
        # A regular hexagon has 120° internal angles
        hexagon = Point(0, 0).buffer(50, resolution=6)
        result = add_dogbone_relief(hexagon, radius_mm=1.6, angle_threshold_deg=100.0)
        # Area should be unchanged (or very close)
        assert result.area == pytest.approx(hexagon.area, rel=0.05)

    def test_returns_polygon(self):
        """Result should always be a Polygon."""
        rect = Polygon([(0, 0), (50, 0), (50, 30), (0, 30)])
        result = add_dogbone_relief(rect, radius_mm=1.6)
        assert result.geom_type == "Polygon"
