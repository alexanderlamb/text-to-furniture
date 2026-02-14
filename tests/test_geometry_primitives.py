"""Tests for geometry_primitives module."""
import numpy as np
import pytest
from shapely.geometry import Polygon

from geometry_primitives import (
    PartProfile2D,
    PartFeature,
    FeatureType,
    polygon_to_profile,
    profile_to_polygon,
    compute_obb_2d,
    project_faces_to_2d,
)


class TestPolygonConversions:
    """Test Shapely <-> Component.profile conversions."""

    def test_polygon_to_profile_rectangle(self):
        poly = Polygon([(10, 20), (110, 20), (110, 70), (10, 70)])
        profile = polygon_to_profile(poly)
        assert len(profile) == 4
        # Should be translated so min corner is at origin
        xs = [p[0] for p in profile]
        ys = [p[1] for p in profile]
        assert min(xs) == pytest.approx(0.0)
        assert min(ys) == pytest.approx(0.0)
        assert max(xs) == pytest.approx(100.0)
        assert max(ys) == pytest.approx(50.0)

    def test_profile_to_polygon_roundtrip(self):
        original = [(0, 0), (100, 0), (100, 50), (0, 50)]
        poly = profile_to_polygon(original)
        assert poly.area == pytest.approx(5000.0)
        profile = polygon_to_profile(poly)
        assert len(profile) == 4

    def test_empty_profile(self):
        poly = profile_to_polygon([])
        assert poly.is_empty

    def test_polygon_to_profile_empty(self):
        poly = Polygon()
        profile = polygon_to_profile(poly)
        assert profile == []


class TestComputeOBB:
    """Test oriented bounding box computation."""

    def test_axis_aligned_rectangle(self):
        poly = Polygon([(0, 0), (200, 0), (200, 100), (0, 100)])
        w, h, angle = compute_obb_2d(poly)
        assert w == pytest.approx(200.0, abs=1.0)
        assert h == pytest.approx(100.0, abs=1.0)

    def test_square(self):
        poly = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
        w, h, angle = compute_obb_2d(poly)
        assert w == pytest.approx(50.0, abs=1.0)
        assert h == pytest.approx(50.0, abs=1.0)

    def test_empty_polygon(self):
        poly = Polygon()
        w, h, angle = compute_obb_2d(poly)
        assert w == 0.0
        assert h == 0.0


class TestPartProfile2D:
    """Test PartProfile2D validation."""

    def test_valid_profile(self, simple_rectangle_profile):
        issues = simple_rectangle_profile.validate_geometry()
        assert issues == []

    def test_empty_outline(self):
        profile = PartProfile2D(outline=Polygon(), thickness_mm=6.35)
        issues = profile.validate_geometry()
        assert any("empty" in i.lower() for i in issues)

    def test_cutout_outside_outline(self):
        outline = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        cutout = Polygon([(200, 200), (300, 200), (300, 300), (200, 300)])
        profile = PartProfile2D(
            outline=outline,
            cutouts=[cutout],
            thickness_mm=6.35,
        )
        issues = profile.validate_geometry()
        assert any("outside" in i.lower() for i in issues)

    def test_net_polygon(self, simple_rectangle_profile):
        net = simple_rectangle_profile.net_polygon()
        assert net.area == pytest.approx(200 * 100, abs=1.0)

    def test_net_polygon_with_cutout(self):
        outline = Polygon([(0, 0), (200, 0), (200, 100), (0, 100)])
        cutout = Polygon([(50, 25), (150, 25), (150, 75), (50, 75)])
        profile = PartProfile2D(
            outline=outline,
            cutouts=[cutout],
            thickness_mm=6.35,
        )
        net = profile.net_polygon()
        expected = 200 * 100 - 100 * 50
        assert net.area == pytest.approx(expected, abs=1.0)


class TestProjectFaces:
    """Test face projection to 2D."""

    def test_project_box_top(self, box_mesh):
        mesh = box_mesh
        normals = mesh.face_normals
        # Find faces with normal pointing up (Z+)
        up_faces = [i for i in range(len(normals)) if normals[i][2] > 0.9]
        assert len(up_faces) > 0

        poly = project_faces_to_2d(
            mesh, up_faces,
            plane_normal=np.array([0, 0, 1.0]),
            plane_offset=100.0,
        )
        assert not poly.is_empty
        assert poly.area > 0
