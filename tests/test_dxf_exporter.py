"""Tests for dxf_exporter module."""
import os
import tempfile

import pytest
from shapely.geometry import Polygon

from geometry_primitives import PartProfile2D
from dxf_exporter import (
    DXFExportConfig,
    part_to_dxf,
    design_to_dxf,
    design_to_nested_dxf,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestPartToDXF:
    """Test single part DXF export."""

    def test_exports_rectangle(self, tmp_dir):
        """A simple rectangle should produce a valid DXF."""
        outline = Polygon([(0, 0), (200, 0), (200, 100), (0, 100)])
        profile = PartProfile2D(outline=outline, thickness_mm=6.35)

        filepath = os.path.join(tmp_dir, "rect.dxf")
        result = part_to_dxf(profile, filepath, part_name="test_rect")

        assert result == filepath
        assert os.path.isfile(filepath)
        assert os.path.getsize(filepath) > 0

    def test_exports_with_cutouts(self, tmp_dir):
        """Part with cutouts should export both outline and cutouts."""
        outline = Polygon([(0, 0), (200, 0), (200, 100), (0, 100)])
        cutout = Polygon([(50, 30), (70, 30), (70, 70), (50, 70)])
        profile = PartProfile2D(
            outline=outline,
            cutouts=[cutout],
            thickness_mm=6.35,
        )

        filepath = os.path.join(tmp_dir, "with_cutout.dxf")
        result = part_to_dxf(profile, filepath, part_name="test_cutout")

        assert os.path.isfile(filepath)
        # File with cutout should be larger than minimal
        assert os.path.getsize(filepath) > 100


class TestDesignToDXF:
    """Test multi-part DXF export."""

    def test_exports_multiple_parts(self, tmp_dir):
        """Multiple parts should each get a separate DXF file."""
        parts = [
            ("part_0", PartProfile2D(
                outline=Polygon([(0, 0), (200, 0), (200, 100), (0, 100)]),
                thickness_mm=6.35,
            )),
            ("part_1", PartProfile2D(
                outline=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
                thickness_mm=6.35,
            )),
        ]

        output_dir = os.path.join(tmp_dir, "parts")
        paths = design_to_dxf(parts, output_dir)

        assert len(paths) == 2
        for path in paths:
            assert os.path.isfile(path)


class TestNestedDXF:
    """Test nested sheet layout DXF export."""

    def test_nested_layout(self, tmp_dir):
        """Nested layout should produce a single DXF with all parts."""
        parts = [
            ("part_0", PartProfile2D(
                outline=Polygon([(0, 0), (200, 0), (200, 100), (0, 100)]),
                thickness_mm=6.35,
            )),
            ("part_1", PartProfile2D(
                outline=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
                thickness_mm=6.35,
            )),
        ]

        filepath = os.path.join(tmp_dir, "nested.dxf")
        result = design_to_nested_dxf(parts, filepath)

        assert os.path.isfile(filepath)
        assert os.path.getsize(filepath) > 0
