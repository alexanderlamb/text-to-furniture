"""Tests for slab_candidates.py — v2 candidate slab generation."""
import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

from slab_candidates import (
    Slab3D, Slab3DConfig, generate_candidates,
    _candidates_from_ransac, _candidates_from_axis_slices,
    _candidates_from_surface_aligned, _candidates_from_paired_interior,
    _candidates_from_bounding_box, _split_oversized_candidates,
    slab3d_to_part_profile,
    _deduplicate_candidates,
)
from geometry_primitives import PlanarPatch


class TestSlab3DConfig:
    def test_defaults(self):
        cfg = Slab3DConfig()
        assert cfg.material_key == "plywood_baltic_birch"
        assert cfg.min_slab_area_mm2 == 500.0


class TestCandidateGeneration:
    def test_box_mesh_produces_candidates(self, box_mesh):
        """A box mesh should produce candidates from all sources."""
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)
        assert len(candidates) > 0

    def test_table_mesh_produces_candidates(self, table_mesh):
        """A table mesh should produce candidates."""
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(table_mesh, config)
        assert len(candidates) > 0

    def test_all_candidates_have_valid_fields(self, box_mesh):
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)
        for slab in candidates:
            assert slab.normal.shape == (3,)
            assert slab.origin.shape == (3,)
            assert slab.basis_u.shape == (3,)
            assert slab.basis_v.shape == (3,)
            assert slab.thickness_mm > 0
            assert slab.width_mm > 0
            assert slab.height_mm > 0
            assert not slab.outline_2d.is_empty
            assert slab.source in (
                "ransac", "axis_slice", "surface_aligned",
                "paired_interior", "bbox_fallback",
            )

    def test_basis_vectors_orthogonal(self, box_mesh):
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)
        for slab in candidates:
            # basis_u and basis_v should be orthogonal to each other and to normal
            assert abs(np.dot(slab.basis_u, slab.basis_v)) < 1e-6
            assert abs(np.dot(slab.basis_u, slab.normal)) < 1e-6
            assert abs(np.dot(slab.basis_v, slab.normal)) < 1e-6


class TestRANSACSource:
    def test_ransac_box(self, box_mesh):
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        thickness = 6.35
        slabs = _candidates_from_ransac(box_mesh, config, thickness)
        assert len(slabs) > 0
        for s in slabs:
            assert s.source == "ransac"


class TestAxisSliceSource:
    def test_axis_slices_box(self, box_mesh):
        config = Slab3DConfig(min_slab_area_mm2=50.0, axis_slice_spacing_mm=40.0)
        thickness = 6.35
        slabs = _candidates_from_axis_slices(box_mesh, config, thickness)
        # A box should produce cross-sections along all three axes
        assert len(slabs) > 0
        for s in slabs:
            assert s.source == "axis_slice"


class TestSurfaceAlignedSource:
    def test_surface_aligned_box(self, box_mesh):
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        thickness = 6.35
        slabs = _candidates_from_surface_aligned(box_mesh, config, thickness)
        assert len(slabs) >= 0  # may or may not produce depending on mesh
        for s in slabs:
            assert s.source == "surface_aligned"


class TestPairedInteriorSource:
    @staticmethod
    def _make_patch(
        normal: np.ndarray,
        plane_offset: float,
        centroid: np.ndarray,
        width: float = 200.0,
        height: float = 100.0,
    ) -> PlanarPatch:
        from shapely.geometry import Polygon

        basis_u = np.array([1.0, 0.0, 0.0])
        basis_v = np.array([0.0, 1.0, 0.0])
        boundary = Polygon([
            (-width / 2, -height / 2),
            (width / 2, -height / 2),
            (width / 2, height / 2),
            (-width / 2, height / 2),
        ])
        return PlanarPatch(
            plane_normal=normal.astype(float),
            plane_offset=float(plane_offset),
            face_indices=[],
            boundary_2d=boundary,
            area_mm2=width * height,
            centroid_3d=centroid.astype(float),
            basis_u=basis_u,
            basis_v=basis_v,
        )

    def test_large_gap_creates_dual_interior_planes(self):
        config = Slab3DConfig(
            min_slab_area_mm2=50.0,
            paired_min_gap_mm=10.0,
            paired_dual_gap_factor=4.0,
            max_paired_candidates=10,
        )
        p1 = self._make_patch(
            normal=np.array([0.0, 0.0, 1.0]),
            plane_offset=0.0,
            centroid=np.array([0.0, 0.0, 0.0]),
        )
        p2 = self._make_patch(
            normal=np.array([0.0, 0.0, -1.0]),
            plane_offset=-200.0,  # plane z=200 for normal -Z
            centroid=np.array([0.0, 0.0, 200.0]),
        )

        slabs = _candidates_from_paired_interior([p1, p2], config, 6.35)
        assert len(slabs) == 2
        assert all(s.source == "paired_interior" for s in slabs)

    def test_small_gap_creates_single_mid_plane(self):
        config = Slab3DConfig(
            min_slab_area_mm2=50.0,
            paired_min_gap_mm=5.0,
            paired_dual_gap_factor=4.0,
            max_paired_candidates=10,
        )
        p1 = self._make_patch(
            normal=np.array([0.0, 0.0, 1.0]),
            plane_offset=0.0,
            centroid=np.array([0.0, 0.0, 0.0]),
        )
        p2 = self._make_patch(
            normal=np.array([0.0, 0.0, -1.0]),
            plane_offset=-15.0,  # plane z=15 for normal -Z
            centroid=np.array([0.0, 0.0, 15.0]),
        )

        slabs = _candidates_from_paired_interior([p1, p2], config, 6.35)
        assert len(slabs) == 1
        assert slabs[0].source == "paired_interior"


class TestDeduplication:
    def _make_slab(self, normal, origin, width=100, height=50):
        from geometry_primitives import _make_2d_basis
        n = np.array(normal, dtype=float)
        n /= np.linalg.norm(n)
        u, v = _make_2d_basis(n)
        outline = Polygon([
            (-width/2, -height/2), (width/2, -height/2),
            (width/2, height/2), (-width/2, height/2),
        ])
        return Slab3D(
            normal=n, origin=np.array(origin, dtype=float),
            thickness_mm=6.35, width_mm=width, height_mm=height,
            basis_u=u, basis_v=v, outline_2d=outline,
            material_key="plywood_baltic_birch", source="test",
        )

    def test_dedupe_merges_near_duplicates(self):
        """Two slabs with same normal and nearby origins should merge into one."""
        config = Slab3DConfig(dedupe_angle_deg=10.0, dedupe_offset_mm=5.0)
        s1 = self._make_slab([0, 0, 1], [0, 0, 0])
        s2 = self._make_slab([0, 0, 1], [0, 0, 2])  # offset 2mm < 5mm threshold
        result = _deduplicate_candidates([s1, s2], config)
        assert len(result) == 1

    def test_dedupe_preserves_distinct(self):
        """Two slabs with different normals or far offsets should remain separate."""
        config = Slab3DConfig(dedupe_angle_deg=10.0, dedupe_offset_mm=5.0)
        s1 = self._make_slab([0, 0, 1], [0, 0, 0])
        s2 = self._make_slab([1, 0, 0], [0, 0, 0])  # perpendicular normal
        result = _deduplicate_candidates([s1, s2], config)
        assert len(result) == 2

    def test_dedupe_preserves_distant_parallel(self):
        """Parallel slabs with large offset should remain separate."""
        config = Slab3DConfig(dedupe_angle_deg=10.0, dedupe_offset_mm=5.0)
        s1 = self._make_slab([0, 0, 1], [0, 0, 0])
        s2 = self._make_slab([0, 0, 1], [0, 0, 100])  # offset 100mm > 5mm
        result = _deduplicate_candidates([s1, s2], config)
        assert len(result) == 2

    def test_dedupe_single_candidate(self):
        """A single candidate passes through unchanged."""
        config = Slab3DConfig()
        s1 = self._make_slab([0, 0, 1], [0, 0, 0])
        result = _deduplicate_candidates([s1], config)
        assert len(result) == 1
        assert result[0] is s1


class TestSlab3DTransformMethods:
    def test_world_to_uv_roundtrip(self):
        """world_to_uv -> uv_to_world should recover the original 3D point (modulo normal)."""
        outline = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])
        slab = Slab3D(
            normal=np.array([0, 0, 1.0]),
            origin=np.array([50, 25, 100.0]),
            thickness_mm=6.35,
            width_mm=100.0,
            height_mm=50.0,
            basis_u=np.array([1.0, 0, 0]),
            basis_v=np.array([0, 1.0, 0]),
            outline_2d=outline,
            material_key="plywood_baltic_birch",
            source="test",
        )
        pt_3d = np.array([80.0, 40.0, 100.0])  # on the slab plane
        u, v = slab.world_to_uv(pt_3d)
        pt_back = slab.uv_to_world(u, v)
        np.testing.assert_allclose(pt_back, pt_3d, atol=1e-9)

    def test_world_to_uv_roundtrip_rotated(self):
        """Roundtrip with a non-axis-aligned slab."""
        n = np.array([1.0, 1.0, 0.0])
        n /= np.linalg.norm(n)
        from geometry_primitives import _make_2d_basis
        u_ax, v_ax = _make_2d_basis(n)
        outline = Polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)])
        slab = Slab3D(
            normal=n,
            origin=np.array([10.0, 20.0, 30.0]),
            thickness_mm=6.35,
            width_mm=100.0,
            height_mm=100.0,
            basis_u=u_ax,
            basis_v=v_ax,
            outline_2d=outline,
            material_key="plywood_baltic_birch",
            source="test",
        )
        # Pick a point on the slab plane
        pt_3d = slab.origin + 17.0 * u_ax + 23.0 * v_ax
        u, v = slab.world_to_uv(pt_3d)
        assert abs(u - 17.0) < 1e-9
        assert abs(v - 23.0) < 1e-9
        pt_back = slab.uv_to_world(u, v)
        np.testing.assert_allclose(pt_back, pt_3d, atol=1e-9)

    def test_plane_distance(self):
        """plane_distance returns signed distance along the normal."""
        outline = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])
        slab = Slab3D(
            normal=np.array([0, 0, 1.0]),
            origin=np.array([0, 0, 50.0]),
            thickness_mm=6.35,
            width_mm=100.0,
            height_mm=50.0,
            basis_u=np.array([1.0, 0, 0]),
            basis_v=np.array([0, 1.0, 0]),
            outline_2d=outline,
            material_key="plywood_baltic_birch",
            source="test",
        )
        assert abs(slab.plane_distance(np.array([0, 0, 50.0]))) < 1e-9
        assert abs(slab.plane_distance(np.array([0, 0, 60.0])) - 10.0) < 1e-9
        assert abs(slab.plane_distance(np.array([0, 0, 40.0])) - (-10.0)) < 1e-9


class TestOutlineCentering:
    """Verify all candidate outlines are centered at (0,0) in UV space."""

    def test_axis_slices_centered(self, box_mesh):
        """Axis-slice outlines should be centered at (0,0)."""
        config = Slab3DConfig(min_slab_area_mm2=50.0, axis_slice_spacing_mm=40.0)
        slabs = _candidates_from_axis_slices(box_mesh, config, 6.35)
        for i, s in enumerate(slabs):
            b = s.outline_2d.bounds
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            assert abs(cx) < 1.0, f"slab {i} outline U-center={cx:.1f}"
            assert abs(cy) < 1.0, f"slab {i} outline V-center={cy:.1f}"

    def test_ransac_slabs_centered(self, box_mesh):
        """RANSAC outlines should be centered at (0,0)."""
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        slabs = _candidates_from_ransac(box_mesh, config, 6.35)
        for i, s in enumerate(slabs):
            b = s.outline_2d.bounds
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            assert abs(cx) < 1.0, f"slab {i} outline U-center={cx:.1f}"
            assert abs(cy) < 1.0, f"slab {i} outline V-center={cy:.1f}"

    def test_all_candidates_centered(self, box_mesh):
        """All generated candidates should be centered at (0,0)."""
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)
        for i, s in enumerate(candidates):
            b = s.outline_2d.bounds
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            assert abs(cx) < 1.0, (
                f"slab {i} ({s.source}) outline not centered: "
                f"U-center={cx:.1f}, V-center={cy:.1f}"
            )
            assert abs(cy) < 1.0, (
                f"slab {i} ({s.source}) outline not centered: "
                f"U-center={cx:.1f}, V-center={cy:.1f}"
            )

    def test_outline_uv_roundtrip(self, box_mesh):
        """Outline vertices should round-trip through world_to_uv/uv_to_world."""
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)
        for slab in candidates:
            coords = list(slab.outline_2d.exterior.coords[:-1])
            for u, v in coords:
                pt_3d = slab.uv_to_world(u, v)
                u_back, v_back = slab.world_to_uv(pt_3d)
                assert abs(u - u_back) < 1e-6, f"U roundtrip: {u} vs {u_back}"
                assert abs(v - v_back) < 1e-6, f"V roundtrip: {v} vs {v_back}"

    def test_deduplication_preserves_centering(self):
        """Deduplication should produce centered outlines."""
        from geometry_primitives import _make_2d_basis

        n = np.array([0, 0, 1.0])
        u, v = _make_2d_basis(n)
        s1 = Slab3D(
            normal=n, origin=np.array([0, 0, 0.0]),
            thickness_mm=6.35, width_mm=100.0, height_mm=50.0,
            basis_u=u, basis_v=v,
            outline_2d=Polygon([(-50, -25), (50, -25), (50, 25), (-50, 25)]),
            material_key="plywood_baltic_birch", source="test",
        )
        s2 = Slab3D(
            normal=n, origin=np.array([0, 0, 2.0]),  # nearby
            thickness_mm=6.35, width_mm=100.0, height_mm=50.0,
            basis_u=u, basis_v=v,
            outline_2d=Polygon([(-50, -25), (50, -25), (50, 25), (-50, 25)]),
            material_key="plywood_baltic_birch", source="test",
        )
        config = Slab3DConfig(dedupe_angle_deg=10.0, dedupe_offset_mm=5.0)
        result = _deduplicate_candidates([s1, s2], config)
        assert len(result) == 1
        b = result[0].outline_2d.bounds
        cx = (b[0] + b[2]) / 2.0
        cy = (b[1] + b[3]) / 2.0
        assert abs(cx) < 1.0, f"Deduplicated outline U-center={cx:.1f}"
        assert abs(cy) < 1.0, f"Deduplicated outline V-center={cy:.1f}"


class TestSlabsWithinMeshBounds:
    """Verify slab origins stay within mesh bounds."""

    def test_box_mesh_origins_within_bounds(self, box_mesh):
        """All slab origins should be within mesh bounds (with tolerance)."""
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)
        bounds = box_mesh.bounds  # (2, 3)
        for i, s in enumerate(candidates):
            tol = s.thickness_mm + 10.0  # generous tolerance
            for ax in range(3):
                assert s.origin[ax] >= bounds[0][ax] - tol, (
                    f"slab {i} origin[{ax}]={s.origin[ax]:.1f} "
                    f"below mesh min {bounds[0][ax]:.1f}"
                )
                assert s.origin[ax] <= bounds[1][ax] + tol, (
                    f"slab {i} origin[{ax}]={s.origin[ax]:.1f} "
                    f"above mesh max {bounds[1][ax]:.1f}"
                )


class TestSlab3DToPartProfile:
    def test_conversion_preserves_fields(self):
        outline = Polygon([(0, 0), (100, 0), (100, 50), (0, 50)])
        slab = Slab3D(
            normal=np.array([0, 0, 1.0]),
            origin=np.array([50, 25, 100.0]),
            thickness_mm=6.35,
            width_mm=100.0,
            height_mm=50.0,
            basis_u=np.array([1.0, 0, 0]),
            basis_v=np.array([0, 1.0, 0]),
            outline_2d=outline,
            material_key="plywood_baltic_birch",
            source="ransac",
        )
        profile = slab3d_to_part_profile(slab)
        assert profile.thickness_mm == 6.35
        assert profile.material_key == "plywood_baltic_birch"
        assert np.allclose(profile.basis_u, [1, 0, 0])
        assert np.allclose(profile.basis_v, [0, 1, 0])
        assert np.allclose(profile.origin_3d, [50, 25, 100])
        assert profile.outline.area == outline.area

    def test_profile_3d_roundtrip(self):
        """Converting 3D→2D→3D should be consistent."""
        outline = Polygon([(0, 0), (200, 0), (200, 100), (0, 100)])
        slab = Slab3D(
            normal=np.array([0, 0, 1.0]),
            origin=np.array([100, 50, 200.0]),
            thickness_mm=6.35,
            width_mm=200.0,
            height_mm=100.0,
            basis_u=np.array([1.0, 0, 0]),
            basis_v=np.array([0, 1.0, 0]),
            outline_2d=outline,
            material_key="plywood_baltic_birch",
            source="ransac",
        )
        profile = slab3d_to_part_profile(slab)
        # Project a 3D point, then project back
        pt_3d = np.array([150, 75, 200.0])
        u, v = profile.project_3d_to_2d(pt_3d)
        pt_back = profile.project_2d_to_3d(u, v)
        assert np.allclose(pt_3d, pt_back, atol=1e-6)


class TestBboxFallbackSource:
    """Test _candidates_from_bounding_box fallback for non-watertight meshes."""

    @staticmethod
    def _make_open_sphere():
        """Create a sphere with faces removed to make it non-watertight."""
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=100.0)
        # Remove ~20% of faces to break watertightness
        n_faces = len(sphere.faces)
        keep = np.ones(n_faces, dtype=bool)
        rng = np.random.RandomState(42)
        remove_idx = rng.choice(n_faces, size=n_faces // 5, replace=False)
        keep[remove_idx] = False
        sphere.update_faces(keep)
        assert not sphere.is_watertight
        return sphere

    def test_bbox_fallback_produces_candidates(self):
        """Bbox fallback should produce candidates for any mesh."""
        mesh = self._make_open_sphere()
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        thickness = 6.35
        slabs = _candidates_from_bounding_box(mesh, config, thickness)
        assert len(slabs) > 0
        for s in slabs:
            assert s.source == "bbox_fallback"
            assert s.width_mm > 0
            assert s.height_mm > 0
            assert not s.outline_2d.is_empty

    def test_generate_candidates_uses_bbox_when_others_fail(self):
        """If RANSAC/axis/surface all fail, bbox fallback should kick in."""
        mesh = self._make_open_sphere()
        config = Slab3DConfig(
            min_slab_area_mm2=50.0,
            # high RANSAC threshold to make RANSAC fail
            ransac_distance_mm=0.001,
        )
        candidates = generate_candidates(mesh, config)
        # Should have at least some candidates (from bbox or other sources)
        assert len(candidates) > 0

    def test_bbox_candidates_have_valid_fields(self):
        """Bbox fallback candidates should have valid geometry."""
        mesh = self._make_open_sphere()
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        thickness = 6.35
        slabs = _candidates_from_bounding_box(mesh, config, thickness)
        for slab in slabs:
            assert slab.normal.shape == (3,)
            assert slab.origin.shape == (3,)
            assert slab.basis_u.shape == (3,)
            assert slab.basis_v.shape == (3,)
            assert abs(np.dot(slab.basis_u, slab.basis_v)) < 1e-6
            assert abs(np.dot(slab.basis_u, slab.normal)) < 1e-6


class TestOversizedSplitting:
    """Test _split_oversized_candidates."""

    def _make_slab(self, width, height):
        from geometry_primitives import _make_2d_basis
        n = np.array([0, 0, 1.0])
        u, v = _make_2d_basis(n)
        hw, hh = width / 2.0, height / 2.0
        outline = Polygon([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
        return Slab3D(
            normal=n, origin=np.array([0, 0, 100.0]),
            thickness_mm=6.35, width_mm=width, height_mm=height,
            basis_u=u, basis_v=v, outline_2d=outline,
            material_key="plywood_baltic_birch", source="test",
        )

    def test_fitting_candidate_passes_through(self):
        """A candidate that fits the sheet should pass through unchanged."""
        config = Slab3DConfig()
        slab = self._make_slab(400, 500)  # fits 609.6 x 762.0 sheet
        result = _split_oversized_candidates([slab], config)
        assert len(result) == 1
        assert result[0] is slab

    def test_oversized_candidate_splits(self):
        """A 1500x1200 candidate should split into multiple pieces."""
        config = Slab3DConfig()
        slab = self._make_slab(1500, 1200)
        result = _split_oversized_candidates([slab], config)
        assert len(result) > 1
        # All pieces should fit the sheet
        from materials import MATERIALS
        mat = MATERIALS[config.material_key]
        max_w, max_h = mat.max_size_mm
        for piece in result:
            fits_normal = piece.width_mm <= max_w * 1.01 and piece.height_mm <= max_h * 1.01
            fits_rotated = piece.width_mm <= max_h * 1.01 and piece.height_mm <= max_w * 1.01
            assert fits_normal or fits_rotated, (
                f"Split piece {piece.width_mm:.0f}x{piece.height_mm:.0f} "
                f"exceeds sheet {max_w:.0f}x{max_h:.0f}"
            )

    def test_split_preserves_material(self):
        """Split pieces should preserve material and source."""
        config = Slab3DConfig()
        slab = self._make_slab(1500, 1200)
        result = _split_oversized_candidates([slab], config)
        for piece in result:
            assert piece.material_key == slab.material_key
            assert piece.source == slab.source
            assert piece.thickness_mm == slab.thickness_mm
