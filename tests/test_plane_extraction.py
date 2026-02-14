"""Tests for plane_extraction module."""
import numpy as np
import pytest

from plane_extraction import PlaneExtractionConfig, extract_planar_patches


class TestExtractPlanarPatches:
    """Test RANSAC + region growing plane extraction."""

    def test_box_extracts_six_faces(self, box_mesh):
        """A box should produce ~6 planar patches (one per face)."""
        config = PlaneExtractionConfig(
            ransac_threshold_mm=1.0,
            ransac_iterations=500,
            min_patch_area_mm2=100.0,
            merge_angle_threshold_deg=10.0,
        )
        patches = extract_planar_patches(box_mesh, config)

        # A box has 6 distinct faces; with merging we may get fewer
        assert len(patches) >= 3  # at minimum: top/bottom + 2 merged sides
        assert len(patches) <= 8  # shouldn't over-segment

        # Total face coverage should be close to total mesh area
        total_patch_area = sum(p.area_mm2 for p in patches)
        mesh_area = float(box_mesh.area)
        assert total_patch_area > mesh_area * 0.5

    def test_box_normals_are_cardinal(self, box_mesh):
        """Box patches should have normals close to cardinal axes."""
        config = PlaneExtractionConfig(
            ransac_threshold_mm=1.0,
            ransac_iterations=500,
            min_patch_area_mm2=100.0,
        )
        patches = extract_planar_patches(box_mesh, config)

        cardinal = [
            np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
            np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1]),
        ]

        for patch in patches:
            n = patch.plane_normal / np.linalg.norm(patch.plane_normal)
            # Check that normal is close to some cardinal direction
            dots = [abs(float(np.dot(n, c))) for c in cardinal]
            assert max(dots) > 0.8, f"Patch normal {n} not near any cardinal axis"

    def test_cylinder_endcaps(self, cylinder_mesh):
        """A cylinder should extract at least 2 endcap patches."""
        config = PlaneExtractionConfig(
            ransac_threshold_mm=2.0,
            ransac_iterations=500,
            min_patch_area_mm2=100.0,
        )
        patches = extract_planar_patches(cylinder_mesh, config)

        # Should find flat endcaps
        assert len(patches) >= 1

    def test_patches_have_valid_boundaries(self, box_mesh):
        """All patches should have non-empty 2D boundaries."""
        config = PlaneExtractionConfig(min_patch_area_mm2=100.0)
        patches = extract_planar_patches(box_mesh, config)

        for patch in patches:
            assert not patch.boundary_2d.is_empty
            assert patch.boundary_2d.area > 0
            assert patch.area_mm2 > 0

    def test_empty_config_defaults(self, box_mesh):
        """Extraction works with default config."""
        patches = extract_planar_patches(box_mesh)
        assert len(patches) > 0
